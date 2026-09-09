defmodule MLServe.ModelServer do
  @moduledoc """
  Owns one model version's lifecycle: loading, readiness, status and draining.

  ## Loading never blocks the supervisor

  `init/1` returns immediately with status `:loading` and defers the actual work to
  `handle_continue/2`. A model that takes thirty seconds to memory-map from disk would otherwise
  hold up `Supervisor.start_link/2` for thirty seconds, and with several such models an
  application could exceed its own start timeout and fail to boot at all.

  Until loading finishes, `MLServe.predict/3` returns `{:error, :model_not_ready}` and
  `MLServe.ready?/1` returns `false` — which is exactly what a Kubernetes readiness probe should
  see. Use `MLServe.await_ready/2` when you need to block.

  ## Load failures retry with backoff

  A failed load does not crash the process. Model artifacts live on network mounts, object-store
  fuse layers and volumes that attach a moment after the container starts; crashing would burn
  the supervisor's restart intensity in seconds and take down the whole instance permanently.
  Instead the load is retried with exponential backoff, and after the final attempt the model is
  marked `:failed` with the reason preserved in `MLServe.model_status/2`.

  ## Draining

  Unload is graceful. The model is marked `:draining` so the registry stops routing new requests
  to it, then this process waits for the in-flight counter to reach zero before terminating the
  workers. Requests already accepted finish; requests not yet accepted go elsewhere. Only after
  `:drain_timeout` elapses are stragglers abandoned, and the count that was still outstanding is
  reported as the `drained` measurement on `[:ml_serve, :model, :unload]`.
  """

  use GenServer, restart: :transient

  require Logger

  alias MLServe.Backend
  alias MLServe.Batcher
  alias MLServe.ModelRegistry
  alias MLServe.ModelSpec
  alias MLServe.Route
  alias MLServe.Security
  alias MLServe.Telemetry
  alias MLServe.WorkerSupervisor

  @max_load_attempts 3
  @base_backoff 200

  defstruct [:spec, :state, :loaded_at, :load_duration, attempt: 0]

  # Client API

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    spec = Keyword.fetch!(opts, :spec)
    GenServer.start_link(__MODULE__, opts, name: via(spec.name, spec.version))
  end

  @doc false
  @spec via(atom(), String.t()) :: GenServer.name()
  def via(name, version) do
    {:via, Registry, {MLServe.Registry, {:model_server, name, version}}}
  end

  @doc false
  @spec whereis(atom(), String.t()) :: pid() | nil
  def whereis(name, version) do
    case Registry.lookup(MLServe.Registry, {:model_server, name, version}) do
      [{pid, _value}] -> pid
      [] -> nil
    end
  end

  @doc """
  Drains in-flight requests and returns the number still outstanding when the wait ended.
  """
  @spec drain(atom(), String.t(), timeout()) :: {:ok, non_neg_integer()} | {:error, term()}
  def drain(name, version, timeout) do
    case whereis(name, version) do
      nil -> {:error, :model_not_found}
      pid -> GenServer.call(pid, {:drain, timeout}, timeout + 1_000)
    end
  catch
    :exit, _ -> {:error, :model_not_found}
  end

  @doc "Returns the loaded backend state. Used by tests and by reload."
  @spec state(atom(), String.t()) :: {:ok, term()} | {:error, :model_not_ready}
  def state(name, version) do
    case whereis(name, version) do
      nil -> {:error, :model_not_ready}
      pid -> GenServer.call(pid, :state)
    end
  catch
    :exit, _ -> {:error, :model_not_ready}
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    spec = Keyword.fetch!(opts, :spec)

    # Required for terminate/2 to run at all. Without trapping exits the supervisor's shutdown
    # signal kills this process outright and the backend's unload/1 never fires, leaking ports,
    # file handles and GPU memory on every unload.
    Process.flag(:trap_exit, true)

    {:ok, %__MODULE__{spec: spec}, {:continue, :load}}
  end

  @impl true
  def handle_continue(:load, %__MODULE__{spec: spec} = server) do
    started_at = System.monotonic_time()

    case load(spec) do
      {:ok, spec, state} ->
        case start_children(spec, state) do
          :ok ->
            duration = System.monotonic_time() - started_at
            publish(spec, state)

            # Emitted before the model is marked :ready, so anything observing readiness — a probe,
            # await_ready/2, a test — can rely on the load event having already fired. The reverse
            # order leaves an observable window where a model serves traffic that no telemetry
            # consumer has been told about.
            Telemetry.model_load(
              %{
                model: spec.name,
                version: spec.version,
                backend: spec.backend,
                workers: ModelSpec.worker_count(spec),
                result: :ok
              },
              duration
            )

            mark_ready(spec, state, duration)

            {:noreply,
             %{
               server
               | spec: spec,
                 state: state,
                 loaded_at: DateTime.utc_now(),
                 load_duration: native_to_ms(duration)
             }}

          {:error, reason} ->
            if state, do: Backend.unload(spec, state)
            retry_or_fail(server, reason, started_at)
        end

      {:error, reason} ->
        retry_or_fail(server, reason, started_at)
    end
  end

  @impl true
  def handle_call({:drain, timeout}, _from, %__MODULE__{spec: spec} = server) do
    ModelRegistry.update_status(spec.name, spec.version, :draining)
    outstanding = await_drain(spec, timeout)
    {:reply, {:ok, outstanding}, server}
  end

  def handle_call(:state, _from, server), do: {:reply, {:ok, server.state}, server}

  @impl true
  def handle_info(:retry_load, server) do
    handle_continue(:load, server)
  end

  def handle_info(_message, server), do: {:noreply, server}

  @impl true
  def terminate(_reason, %__MODULE__{spec: spec, state: state}) do
    :persistent_term.erase(Route.state_key(spec.name, spec.version))
    if state, do: Backend.unload(spec, state)
    :ok
  end

  # Private Functions

  # With load: :per_worker the workers each call the backend themselves, so loading centrally
  # here as well would load the model N+1 times. The artifact is still validated first, and the
  # resolved path is threaded back so workers load from the same checked file.
  defp load(spec) do
    with {:ok, spec} <- validate_artifact(spec) do
      case spec.load do
        :per_worker ->
          {:ok, spec, nil}

        :once ->
          case Backend.load(spec) do
            {:ok, state} -> {:ok, spec, state}
            {:error, reason} -> {:error, reason}
          end
      end
    end
  end

  defp validate_artifact(%ModelSpec{path: nil} = spec), do: {:ok, spec}

  defp validate_artifact(%ModelSpec{path: path} = spec) do
    case Security.validate_path(path, checksum: spec.checksum) do
      {:ok, resolved} ->
        {:ok, %{spec | path: resolved, config: Keyword.put(spec.config, :path, resolved)}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Shared-concurrency models publish their state to :persistent_term so that predict/3 can read
  # it in the caller without a copy. Writes trigger a global scan, which is why this happens once
  # per load and never per request.
  defp publish(%ModelSpec{concurrency: :shared} = spec, state) do
    :persistent_term.put(Route.state_key(spec.name, spec.version), {spec, state})
  end

  defp publish(_spec, _state), do: :ok

  defp start_children(%ModelSpec{concurrency: :shared} = spec, _state) do
    start_batcher(spec)
  end

  defp start_children(spec, state) do
    parent = self()

    with {:ok, _pid} <-
           Supervisor.start_child(
             parent_supervisor(parent),
             Supervisor.child_spec({WorkerSupervisor, spec: spec, state: state}, id: :workers)
           ),
         :ok <- verify_workers(spec),
         :ok <- start_batcher(spec) do
      :ok
    else
      {:error, {:already_started, _pid}} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  # A worker whose own load/1 fails stops with :ignore rather than crash-looping, which the
  # supervisor treats as a successful start. Without this check a per-worker backend that cannot
  # load at all would present as a healthy model with an empty pool.
  defp verify_workers(spec) do
    alive =
      Enum.count(0..(ModelSpec.worker_count(spec) - 1)//1, fn index ->
        MLServe.Worker.whereis(spec.name, spec.version, index) != nil
      end)

    if alive > 0, do: :ok, else: {:error, :no_workers_started}
  end

  defp start_batcher(%ModelSpec{batching: nil}), do: :ok

  defp start_batcher(spec) do
    case Supervisor.start_child(
           parent_supervisor(self()),
           Supervisor.child_spec({Batcher, spec: spec}, id: :batcher)
         ) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  # The ModelInstance supervisor is this process's parent. Asking for it rather than storing a
  # name keeps ModelServer usable under any supervisor in tests.
  defp parent_supervisor(pid) do
    {:dictionary, dictionary} = Process.info(pid, :dictionary)
    Keyword.fetch!(dictionary, :"$ancestors") |> hd()
  end

  defp mark_ready(spec, state, duration) do
    ModelRegistry.update_status(spec.name, spec.version, :ready,
      loaded_at: DateTime.utc_now(),
      load_duration_ms: native_to_ms(duration),
      metadata: metadata(spec, state),
      failure: nil
    )
  end

  # With :per_worker there is no central state to introspect; the workers hold it.
  defp metadata(_spec, nil), do: %{}
  defp metadata(spec, state), do: Backend.metadata(spec, state)

  defp retry_or_fail(%__MODULE__{spec: spec, attempt: attempt} = server, reason, started_at) do
    if attempt + 1 < @max_load_attempts do
      delay = backoff(attempt)

      Logger.warning(
        "[ml_serve] #{inspect(spec.name)} v#{spec.version} failed to load " <>
          "(attempt #{attempt + 1}/#{@max_load_attempts}), retrying in #{delay}ms: #{inspect(reason)}"
      )

      Process.send_after(self(), :retry_load, delay)
      {:noreply, %{server | attempt: attempt + 1}}
    else
      duration = System.monotonic_time() - started_at

      Logger.error(
        "[ml_serve] #{inspect(spec.name)} v#{spec.version} failed to load after " <>
          "#{@max_load_attempts} attempts: #{inspect(reason)}"
      )

      ModelRegistry.update_status(spec.name, spec.version, :failed, failure: reason)

      Telemetry.model_load(
        %{
          model: spec.name,
          version: spec.version,
          backend: spec.backend,
          workers: 0,
          result: :error
        },
        duration
      )

      {:noreply, %{server | attempt: attempt + 1}}
    end
  end

  defp await_drain(spec, timeout) do
    deadline = System.monotonic_time(:millisecond) + timeout
    do_await_drain(spec, deadline)
  end

  defp do_await_drain(spec, deadline) do
    case ModelRegistry.peek_route(spec.name, spec.version) do
      {:ok, route} ->
        in_flight = Route.in_flight(route)

        cond do
          in_flight == 0 -> 0
          System.monotonic_time(:millisecond) >= deadline -> in_flight
          true -> do_await_drain_after(spec, deadline)
        end

      {:error, _} ->
        0
    end
  end

  defp do_await_drain_after(spec, deadline) do
    Process.sleep(10)
    do_await_drain(spec, deadline)
  end

  defp backoff(attempt), do: @base_backoff * Integer.pow(2, attempt)

  defp native_to_ms(native), do: System.convert_time_unit(native, :native, :millisecond)
end
