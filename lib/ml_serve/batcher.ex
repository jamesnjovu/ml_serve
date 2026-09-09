defmodule MLServe.Batcher do
  @moduledoc """
  Coalesces concurrent single predictions into one backend batch call.

  Started only for models configured with `batching: [max_size: 16, timeout: 10]`.

  ## Why this exists

  `MLServe.batch_predict/3` helps when *one* caller has many inputs. Dynamic batching helps the
  far more common production shape: many independent callers, each with one input, arriving
  within milliseconds of each other. A GPU that processes 32 rows in barely more time than one
  row is being wasted by a pool that feeds it one row at a time. This process collects arrivals
  into a window and hands the backend a full batch.

  ## The window

  A batch flushes when either trigger fires:

    * `:max_size` inputs have accumulated (`reason: :full`)
    * `:timeout` milliseconds have passed since the *first* input in the batch (`reason: :timeout`)

  Timing from the first input rather than the last bounds the added latency at `:timeout` for
  every caller. A sliding window timed from the last arrival can starve the earliest caller
  indefinitely under steady traffic.

  Watch `[:ml_serve, :batch, :flush]`: a healthy configuration flushes mostly on `:full`. Mostly
  `:timeout` means the window is longer than your arrival rate justifies, and you are adding
  latency for batches that never fill.

  ## Why the batcher never blocks

  Running inference inside `handle_info(:flush, ...)` would stop the batcher accumulating the
  *next* batch for the whole duration of the current one — serialising exactly what it was built
  to parallelise. Instead a flush hands the batch to a task under `MLServe.TaskSupervisor`, which
  calls a worker and replies to every caller with `GenServer.reply/2`.

  ## Backpressure

  In-flight batches are capped at the model's worker count. Beyond that, flushes wait: there is
  no worker free to take them, and queueing more would only build an unbounded backlog of work
  whose callers will have timed out by the time it runs. That cap is the backpressure.
  """

  use GenServer

  require Logger

  alias MLServe.ModelSpec
  alias MLServe.Route
  alias MLServe.Telemetry
  alias MLServe.Worker

  defstruct [
    :spec,
    :max_size,
    :timeout,
    :max_in_flight,
    :timer,
    :opened_at,
    queue: [],
    size: 0,
    in_flight: 0,
    running: %{}
  ]

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
    {:via, Registry, {MLServe.Registry, {:batcher, name, version}}}
  end

  @doc """
  Submits one input to the batch window and blocks until its result is ready.
  """
  @spec predict(Route.t(), term(), integer(), timeout()) ::
          {:ok, term(), map()} | {:error, term()}
  def predict(%Route{} = route, input, deadline, timeout) do
    GenServer.call(
      via(route.name, route.version),
      {:predict, input, System.monotonic_time(), deadline},
      timeout
    )
  catch
    # As in MLServe.Worker: catch every exit, so a batcher that dies mid-request returns an error
    # rather than killing the caller.
    :exit, {:timeout, _} -> {:error, :timeout}
    :exit, _reason -> {:error, :model_not_ready}
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    spec = Keyword.fetch!(opts, :spec)

    {:ok,
     %__MODULE__{
       spec: spec,
       max_size: spec.batching.max_size,
       timeout: spec.batching.timeout,
       max_in_flight: max(ModelSpec.worker_count(spec), 1)
     }}
  end

  @impl true
  def handle_call({:predict, input, enqueued_at, deadline}, from, state) do
    state = %{
      state
      | queue: [{from, input, enqueued_at, deadline} | state.queue],
        size: state.size + 1,
        opened_at: state.opened_at || System.monotonic_time()
    }

    cond do
      state.size >= state.max_size -> {:noreply, flush(state, :full)}
      state.timer != nil -> {:noreply, state}
      true -> {:noreply, %{state | timer: Process.send_after(self(), :flush, state.timeout)}}
    end
  end

  @impl true
  def handle_info(:flush, state) do
    {:noreply, flush(%{state | timer: nil}, :timeout)}
  end

  def handle_info({:batch_done, ref}, state) do
    Process.demonitor(ref, [:flush])
    {:noreply, release(state, ref)}
  end

  # The flush task died without replying — killed, or its supervisor shut down. Without this the
  # in-flight slot leaks (eventually wedging the batcher permanently) and its callers block until
  # their own timeouts instead of being told immediately.
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case Map.fetch(state.running, ref) do
      {:ok, batch} ->
        Logger.warning(fn ->
          "[ml_serve] #{inspect(state.spec.name)} batch task exited: #{inspect(reason)}"
        end)

        reply_all(batch, {:error, :model_not_ready})
        {:noreply, release(state, ref)}

      :error ->
        {:noreply, state}
    end
  end

  def handle_info(_message, state), do: {:noreply, state}

  # Private Functions

  defp flush(%__MODULE__{size: 0} = state, _reason), do: %{state | opened_at: nil}

  defp flush(%__MODULE__{in_flight: in_flight, max_in_flight: max} = state, _reason)
       when in_flight >= max do
    # Every worker is busy. Hold the batch and let :batch_done wake us; re-arming the timer here
    # would just spin.
    state
  end

  defp flush(state, reason) do
    batch = Enum.reverse(state.queue)
    now = System.monotonic_time()
    wait_duration = now - (state.opened_at || now)

    {live, expired} =
      Enum.split_with(batch, fn {_from, _input, _at, deadline} -> deadline > now end)

    Enum.each(expired, fn {from, _input, _at, _deadline} ->
      GenServer.reply(from, {:error, :timeout})
    end)

    state = %{state | queue: [], size: 0, opened_at: nil, timer: cancel(state.timer)}

    case live do
      [] ->
        state

      live ->
        Telemetry.batch_flush(
          %{model: state.spec.name, version: state.spec.version, reason: reason},
          length(live),
          wait_duration
        )

        {ref, batch} = run_batch(state, live)
        %{state | in_flight: state.in_flight + 1, running: Map.put(state.running, ref, batch)}
    end
  end

  defp run_batch(state, batch) do
    parent = self()
    spec = state.spec
    inputs = Enum.map(batch, fn {_from, input, _at, _deadline} -> input end)
    deadline = batch |> Enum.map(fn {_f, _i, _at, deadline} -> deadline end) |> Enum.min()
    timeout = spec.timeout

    {:ok, pid} =
      Task.Supervisor.start_child(MLServe.TaskSupervisor, fn ->
        receive do
          {:go, ref} ->
            reply_all(batch, dispatch(spec, inputs, deadline, timeout))
            send(parent, {:batch_done, ref})
        end
      end)

    # Monitor before releasing the task, so a task that dies immediately is still accounted for.
    ref = Process.monitor(pid)
    send(pid, {:go, ref})

    {ref, batch}
  end

  defp release(state, ref) do
    state = %{state | in_flight: state.in_flight - 1, running: Map.delete(state.running, ref)}

    # A slot freed up. If work accumulated while every worker was busy, send it now rather than
    # waiting for the next arrival or timer tick.
    if state.size > 0 and state.in_flight < state.max_in_flight do
      flush(cancel_timer(state), :full)
    else
      state
    end
  end

  defp dispatch(spec, inputs, deadline, timeout) do
    route_workers = ModelSpec.worker_count(spec)

    worker =
      Enum.find_value(0..max(route_workers - 1, 0), fn index ->
        Worker.whereis(spec.name, spec.version, index)
      end)

    case worker do
      nil ->
        {:error, :model_not_ready}

      pid ->
        case Worker.batch_predict(pid, inputs, deadline, timeout) do
          {:error, :worker_gone} -> {:error, :model_not_ready}
          result -> result
        end
    end
  end

  defp reply_all(batch, {:ok, results, measurements}) when length(results) == length(batch) do
    batch
    |> Enum.zip(results)
    |> Enum.each(fn {{from, _input, enqueued_at, _deadline}, result} ->
      GenServer.reply(
        from,
        {:ok, result,
         %{
           queue_duration: Map.get(measurements, :queue_duration, 0) + queue_wait(enqueued_at),
           inference_duration: Map.get(measurements, :inference_duration, 0)
         }}
      )
    end)
  end

  defp reply_all(batch, {:ok, results, _measurements}) do
    reply_all(
      batch,
      {:error,
       {:invalid_input,
        "backend returned #{length(results)} results for a batch of #{length(batch)}"}}
    )
  end

  defp reply_all(batch, {:error, _} = error) do
    Enum.each(batch, fn {from, _input, _at, _deadline} -> GenServer.reply(from, error) end)
  end

  defp queue_wait(enqueued_at), do: max(System.monotonic_time() - enqueued_at, 0)

  defp cancel_timer(state), do: %{state | timer: cancel(state.timer)}

  defp cancel(nil), do: nil

  defp cancel(timer) do
    Process.cancel_timer(timer)
    nil
  end
end
