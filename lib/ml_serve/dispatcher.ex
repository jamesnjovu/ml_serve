defmodule MLServe.Dispatcher do
  @moduledoc """
  The prediction hot path: route, admit, cache, hook, dispatch, record.

  Everything in this module runs **in the calling process** — a Phoenix controller, an Oban job,
  a `Task`. Nothing here is a GenServer, and that is the point: MLServe adds no serialisation
  point between your request and the model.

  The order of operations is deliberate:

      route lookup (1 ETS read)
        → admission control (atomic counter)
          → telemetry span opens
            → cache lookup
              → preprocess hook
                → dispatch
                → postprocess hook
              → cache write
          → telemetry span closes
        → counters settled

  Cache, hooks and validation all happen *before* dispatch, so a worker is occupied only for the
  duration of actual inference. A `preprocess` hook that queries Postgres for stored features
  runs on the caller's own scheduler time, never on a GPU worker's.

  ## Dispatch strategies

  | Model | Path |
  | ----- | ---- |
  | `concurrency: :shared` | `:persistent_term.get/1` then the backend, in this process. No message passing, no copies. |
  | `batching:` configured | `GenServer.call` to the model's `MLServe.Batcher`, which coalesces concurrent callers. |
  | `concurrency: :exclusive` | Pick a worker, `GenServer.call` it. |
  """

  alias MLServe.Backend
  alias MLServe.Batcher
  alias MLServe.Cache
  alias MLServe.ModelRegistry
  alias MLServe.Route
  alias MLServe.Telemetry
  alias MLServe.Worker

  @doc """
  Runs a single prediction. See `MLServe.predict/3`.
  """
  @spec predict(atom(), term(), keyword()) :: {:ok, term()} | {:error, term()}
  def predict(name, input, opts) do
    with {:ok, route} <- ModelRegistry.route(name, opts) do
      guarded(route, fn ->
        Telemetry.span(metadata(route, false, 1), fn ->
          run_single(route, input, opts)
        end)
      end)
    end
  end

  @doc """
  Runs a batch prediction. See `MLServe.batch_predict/3`.
  """
  @spec batch_predict(atom(), [term()], keyword()) :: {:ok, [term()]} | {:error, term()}
  def batch_predict(name, inputs, opts) when is_list(inputs) do
    with {:ok, route} <- ModelRegistry.route(name, opts),
         :ok <- check_batch_size(route, inputs) do
      guarded(route, fn ->
        Telemetry.span(metadata(route, true, length(inputs)), fn ->
          run_batch(route, inputs, opts)
        end)
      end)
    end
  end

  def batch_predict(_name, inputs, _opts) do
    {:error, {:invalid_input, "batch_predict/3 expects a list, got: #{inspect(inputs)}"}}
  end

  # Admission + counters

  # in_flight must be decremented on every exit path, including a hook raising in user code,
  # or the counter drifts upward and eventually the model refuses all traffic as :overloaded.
  defp guarded(route, fun) do
    case Route.admit(route) do
      :ok ->
        try do
          result = fun.()
          Route.leave(route, match?({:ok, _}, result))
          result
        catch
          kind, reason ->
            Route.leave(route, false)
            :erlang.raise(kind, reason, __STACKTRACE__)
        end

      {:error, :overloaded} ->
        {:error, :overloaded}
    end
  end

  # Single prediction

  defp run_single(route, input, opts) do
    cache = cache_config(route, opts)

    case cache_lookup(route, cache, input, opts) do
      {:ok, cached} ->
        {{:ok, cached}, %{metadata: %{cached?: true}}}

      :miss ->
        case apply_hook(route.preprocess, input) do
          {:error, _} = error ->
            {error, %{}}

          {:ok, prepared} ->
            {result, measurements} = dispatch(route, prepared, opts)

            result
            |> finish(route, cache, input, opts)
            |> then(&{&1, %{measurements: measurements, metadata: %{cached?: false}}})
        end
    end
  end

  defp run_batch(route, inputs, opts) do
    case apply_hooks(route.preprocess, inputs) do
      {:error, _} = error ->
        {error, %{}}

      {:ok, prepared} ->
        {result, measurements} = dispatch_batch(route, prepared, opts)

        result
        |> post_batch(route)
        |> then(&{&1, %{measurements: measurements, metadata: %{cached?: false}}})
    end
  end

  defp finish({:ok, result}, route, cache, input, opts) do
    case apply_hook(route.postprocess, result) do
      {:ok, final} ->
        maybe_cache(route, cache, input, final, opts)
        {:ok, final}

      {:error, _} = error ->
        error
    end
  end

  defp finish(error, _route, _cache, _input, _opts), do: error

  defp post_batch({:ok, results}, route), do: apply_hooks(route.postprocess, results)
  defp post_batch(error, _route), do: error

  # Dispatch

  defp dispatch(%Route{concurrency: :shared} = route, input, _opts) do
    started_at = System.monotonic_time()

    result =
      case shared_state(route) do
        {:ok, spec, state} -> Backend.predict(spec, state, input)
        :error -> {:error, :model_not_ready}
      end

    {result, %{queue_duration: 0, inference_duration: System.monotonic_time() - started_at}}
  end

  defp dispatch(%Route{batching: true} = route, input, opts) do
    timeout = timeout(route, opts)

    case Batcher.predict(route, input, deadline(timeout), timeout) do
      {:ok, result, measurements} -> {{:ok, result}, measurements}
      {:error, _} = error -> {error, %{}}
    end
  end

  defp dispatch(%Route{} = route, input, opts) do
    timeout = timeout(route, opts)
    deadline = deadline(timeout)

    with_worker(route, fn worker -> Worker.predict(worker, input, deadline, timeout) end)
  end

  defp dispatch_batch(%Route{concurrency: :shared} = route, inputs, _opts) do
    started_at = System.monotonic_time()

    result =
      case shared_state(route) do
        {:ok, spec, state} -> Backend.batch_predict(spec, state, inputs)
        :error -> {:error, :model_not_ready}
      end

    {result, %{queue_duration: 0, inference_duration: System.monotonic_time() - started_at}}
  end

  defp dispatch_batch(%Route{} = route, inputs, opts) do
    timeout = timeout(route, opts)
    deadline = deadline(timeout)

    with_worker(route, fn worker -> Worker.batch_predict(worker, inputs, deadline, timeout) end)
  end

  # A worker that dies is deregistered asynchronously, so selection can hand back a pid that is
  # already gone. Retrying with a fresh selection makes a worker restart invisible to callers
  # instead of failing a request the pool could have served. Bounded, so a model whose pool is
  # genuinely empty still fails fast.
  @worker_attempts 3

  defp with_worker(route, fun, attempt \\ 1) do
    case select_worker(route) do
      nil ->
        {{:error, :model_not_ready}, %{}}

      worker ->
        case fun.(worker) do
          {:ok, result, measurements} ->
            {{:ok, result}, measurements}

          {:error, :worker_gone} when attempt < @worker_attempts ->
            with_worker(route, fun, attempt + 1)

          {:error, :worker_gone} ->
            {{:error, :model_not_ready}, %{}}

          {:error, _} = error ->
            {error, %{}}
        end
    end
  end

  # Worker selection

  @doc false
  @spec select_worker(Route.t()) :: pid() | nil
  def select_worker(%Route{workers: 0}), do: nil

  def select_worker(%Route{selection: :round_robin} = route) do
    # A worker restarting leaves a momentary gap in the Registry. Trying a couple of subsequent
    # indices costs two ETS reads and turns a transient :noproc into a served request.
    find_alive(route, Route.next_index(route))
  end

  def select_worker(%Route{selection: :random} = route) do
    find_alive(route, :rand.uniform(route.workers) - 1)
  end

  def select_worker(%Route{selection: :least_loaded} = route) do
    # Power of two choices: sampling two workers and taking the shorter mailbox gets most of the
    # benefit of a full scan at constant cost, and avoids the herd behaviour of always picking
    # the single least-loaded worker.
    a = find_alive(route, :rand.uniform(route.workers) - 1)
    b = find_alive(route, :rand.uniform(route.workers) - 1)

    case {a, b} do
      {nil, nil} -> nil
      {nil, pid} -> pid
      {pid, nil} -> pid
      {pid_a, pid_b} -> if queue_len(pid_a) <= queue_len(pid_b), do: pid_a, else: pid_b
    end
  end

  defp find_alive(route, start_index) do
    Enum.reduce_while(0..min(route.workers - 1, 2), nil, fn offset, _acc ->
      index = rem(start_index + offset, route.workers)

      # Registry deregisters on a monitor message, so a just-killed worker can still be listed.
      # Process.alive?/1 narrows that window; with_worker/3 closes what is left of it.
      case Worker.whereis(route.name, route.version, index) do
        nil -> {:cont, nil}
        pid -> if Process.alive?(pid), do: {:halt, pid}, else: {:cont, nil}
      end
    end)
  end

  defp queue_len(pid) do
    case Process.info(pid, :message_queue_len) do
      {:message_queue_len, len} -> len
      nil -> :infinity
    end
  end

  # Shared state

  defp shared_state(route) do
    case :persistent_term.get(Route.state_key(route.name, route.version), :error) do
      :error -> :error
      {spec, state} -> {:ok, spec, state}
    end
  end

  # Hooks

  defp apply_hook(nil, input), do: {:ok, input}

  defp apply_hook({mod, fun, args}, input), do: normalize_hook(apply(mod, fun, [input | args]))

  defp apply_hook(fun, input) when is_function(fun, 1), do: normalize_hook(fun.(input))

  defp normalize_hook({:ok, _} = ok), do: ok
  defp normalize_hook({:error, {:invalid_input, _}} = error), do: error
  defp normalize_hook({:error, reason}), do: {:error, {:invalid_input, reason}}
  defp normalize_hook(other), do: {:ok, other}

  defp apply_hooks(nil, inputs), do: {:ok, inputs}

  defp apply_hooks(hook, inputs) do
    Enum.reduce_while(inputs, {:ok, []}, fn input, {:ok, acc} ->
      case apply_hook(hook, input) do
        {:ok, value} -> {:cont, {:ok, [value | acc]}}
        {:error, _} = error -> {:halt, error}
      end
    end)
    |> case do
      {:ok, values} -> {:ok, Enum.reverse(values)}
      error -> error
    end
  end

  # Cache

  defp cache_config(route, opts) do
    case Keyword.fetch(opts, :cache) do
      {:ok, true} -> %{enabled: true, ttl: Keyword.get(opts, :cache_ttl, ttl(route))}
      {:ok, false} -> %{enabled: false, ttl: 0}
      {:ok, ttl} when is_integer(ttl) -> %{enabled: true, ttl: ttl}
      :error -> route.cache || %{enabled: false, ttl: 0}
    end
  end

  defp ttl(%Route{cache: %{ttl: ttl}}), do: ttl
  defp ttl(%Route{}), do: MLServe.Config.cache().ttl

  defp cache_lookup(_route, %{enabled: false}, _input, _opts), do: :miss

  defp cache_lookup(route, _cache, input, opts) do
    Cache.fetch(cache_key(route, input, opts), %{model: route.name, version: route.version})
  end

  defp maybe_cache(_route, %{enabled: false}, _input, _result, _opts), do: :ok

  defp maybe_cache(route, %{ttl: ttl}, input, result, opts) do
    Cache.put(cache_key(route, input, opts), result, ttl)
  end

  defp cache_key(route, input, opts) do
    Cache.key(route.name, route.version, Keyword.get(opts, :cache_key, input))
  end

  # Helpers

  defp check_batch_size(route, inputs) do
    if length(inputs) > route.max_batch_size do
      {:error, {:batch_too_large, route.max_batch_size}}
    else
      :ok
    end
  end

  defp timeout(route, opts), do: Keyword.get(opts, :timeout, route.timeout)

  defp deadline(:infinity),
    do: System.monotonic_time() + System.convert_time_unit(1, :second, :native) * 86_400

  defp deadline(timeout) do
    System.monotonic_time() + System.convert_time_unit(timeout, :millisecond, :native)
  end

  defp metadata(route, batch?, batch_size) do
    %{
      model: route.name,
      version: route.version,
      backend: route.backend,
      batch?: batch?,
      canary?: route.canary?,
      batch_size: batch_size
    }
  end
end
