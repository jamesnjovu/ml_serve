defmodule MLServe.Telemetry do
  @moduledoc """
  Telemetry events emitted by MLServe.

  MLServe treats instrumentation as a feature, not an afterthought: `:telemetry` is its only
  runtime dependency, and every prediction is wrapped in a span whether or not anyone is
  listening (an unattached event costs a single ETS lookup).

  Attach `MLServe.Telemetry.Logger.attach/1` for immediate visibility, or
  `MLServe.Telemetry.Metrics.metrics/0` to feed Phoenix LiveDashboard.

  ## Prediction span

  `MLServe.predict/3` and `MLServe.batch_predict/3` emit the standard `:telemetry.span/3` triple.

  ### `[:ml_serve, :prediction, :start]`

  | Measurements | |
  | --- | --- |
  | `system_time` | `System.system_time/0` at the start |
  | `monotonic_time` | `System.monotonic_time/0` at the start |
  | `batch_size` | `1` for `predict/3`, the list length for `batch_predict/3` |

  ### `[:ml_serve, :prediction, :stop]`

  | Measurements | |
  | --- | --- |
  | `duration` | End-to-end, including queueing, hooks and cache |
  | `queue_duration` | Time the request waited before a worker picked it up |
  | `inference_duration` | Time inside the backend callback |
  | `batch_size` | As above |

  `queue_duration` and `inference_duration` are `0` for shared-concurrency backends, which never
  queue — inference runs in the caller.

  ### `[:ml_serve, :prediction, :exception]`

  Measurements `duration`; metadata adds `kind`, `reason` and `stacktrace`. Emitted when the
  backend raises. A backend that *returns* `{:error, reason}` produces a `:stop` event with
  `result: :error` instead — an expected rejection is not an exception.

  ### Prediction metadata

  | Key | |
  | --- | --- |
  | `model` | model name |
  | `version` | the version that actually served the request |
  | `backend` | backend module |
  | `batch?` | whether this came from `batch_predict/3` |
  | `canary?` | whether canary routing chose this version |
  | `cached?` | whether the result came from the cache (`:stop` only) |
  | `result` | `:ok` or `:error` (`:stop` only) |
  | `error_kind` | `:stop` only. `nil` when `result: :ok`; `:raised` when the backend threw and MLServe converted it to a `MLServe.BackendError`; `:returned` when the backend deliberately returned `{:error, reason}`. Alert on `:raised` — that is a bug in the model, not the model doing its job. |

  `version` together with `canary?` is what makes a canary rollout decidable: your metrics
  backend can compare error rate and latency per version without any extra plumbing.

  ## Lifecycle events

    * `[:ml_serve, :model, :load]` — measurements `duration`; metadata `model`, `version`,
      `backend`, `workers`, `result`.
    * `[:ml_serve, :model, :unload]` — measurements `duration`, `drained` (requests still in
      flight when the drain timeout expired; `0` is a clean drain); metadata `model`, `version`,
      `backend`.

  These are single events carrying a duration rather than span triples, because a load either
  happened or it did not — there is no useful window to observe in between.

  ## Cache and batching

    * `[:ml_serve, :cache, :hit]` / `[:ml_serve, :cache, :miss]` — measurements `count: 1`;
      metadata `model`, `version`.
    * `[:ml_serve, :batch, :flush]` — measurements `size`, `wait_duration`; metadata `model`,
      `version`, `reason` (`:full` or `:timeout`). A healthy dynamic-batching setup flushes
      mostly on `:full`; mostly `:timeout` means the batch window is longer than your traffic
      warrants.
  """

  @prediction [:ml_serve, :prediction]
  @model [:ml_serve, :model]
  @cache [:ml_serve, :cache]
  @batch [:ml_serve, :batch]

  @doc """
  Returns every event name MLServe emits.

  Useful for `:telemetry.attach_many/4` and for tests.

  ## Examples

      iex> [:ml_serve, :prediction, :stop] in MLServe.Telemetry.events()
      true
  """
  @spec events() :: [[atom()]]
  def events do
    [
      @prediction ++ [:start],
      @prediction ++ [:stop],
      @prediction ++ [:exception],
      @prediction ++ [:rejected],
      @model ++ [:load],
      @model ++ [:unload],
      @cache ++ [:hit],
      @cache ++ [:miss],
      @batch ++ [:flush]
    ]
  end

  @doc false
  @spec span(map(), (-> {term(), map()})) :: term()
  def span(metadata, fun) do
    start_time = System.monotonic_time()

    :telemetry.execute(
      @prediction ++ [:start],
      %{
        system_time: System.system_time(),
        monotonic_time: start_time,
        batch_size: Map.get(metadata, :batch_size, 1)
      },
      metadata
    )

    try do
      {result, extra} = fun.()

      :telemetry.execute(
        @prediction ++ [:stop],
        Map.merge(
          %{
            duration: System.monotonic_time() - start_time,
            queue_duration: 0,
            inference_duration: 0,
            batch_size: Map.get(metadata, :batch_size, 1)
          },
          Map.get(extra, :measurements, %{})
        ),
        metadata
        |> Map.merge(Map.get(extra, :metadata, %{}))
        |> Map.put(:result, outcome(result))
        |> Map.put(:error_kind, error_kind(result))
      )

      result
    catch
      kind, reason ->
        :telemetry.execute(
          @prediction ++ [:exception],
          %{
            duration: System.monotonic_time() - start_time,
            batch_size: Map.get(metadata, :batch_size, 1)
          },
          Map.merge(metadata, %{kind: kind, reason: reason, stacktrace: __STACKTRACE__})
        )

        :erlang.raise(kind, reason, __STACKTRACE__)
    end
  end

  @doc false
  @spec rejected(map(), pos_integer()) :: :ok
  def rejected(metadata, batch_size) do
    :telemetry.execute(@prediction ++ [:rejected], %{count: 1, batch_size: batch_size}, metadata)
  end

  @doc false
  @spec model_load(map(), non_neg_integer()) :: :ok
  def model_load(metadata, duration) do
    :telemetry.execute(@model ++ [:load], %{duration: duration}, metadata)
  end

  @doc false
  @spec model_unload(map(), non_neg_integer(), non_neg_integer()) :: :ok
  def model_unload(metadata, duration, drained) do
    :telemetry.execute(@model ++ [:unload], %{duration: duration, drained: drained}, metadata)
  end

  @doc false
  @spec cache(:hit | :miss, map()) :: :ok
  def cache(outcome, metadata) when outcome in [:hit, :miss] do
    :telemetry.execute(@cache ++ [outcome], %{count: 1}, metadata)
  end

  @doc false
  @spec batch_flush(map(), pos_integer(), non_neg_integer()) :: :ok
  def batch_flush(metadata, size, wait_duration) do
    :telemetry.execute(@batch ++ [:flush], %{size: size, wait_duration: wait_duration}, metadata)
  end

  defp outcome({:ok, _}), do: :ok
  defp outcome(:ok), do: :ok
  defp outcome(_), do: :error

  # A backend that raised and a backend that returned {:error, reason} both arrive here as an
  # error tuple, but they mean opposite things operationally: the first is a bug in the model,
  # the second is the model doing its job. Only the wrapper tells them apart.
  defp error_kind({:error, {:backend_error, _}}), do: :raised
  defp error_kind({:error, _}), do: :returned
  defp error_kind(_), do: nil
end
