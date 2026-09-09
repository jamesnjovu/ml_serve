defmodule MLServe.Telemetry.Metrics do
  @moduledoc """
  `Telemetry.Metrics` definitions for MLServe, ready for LiveDashboard or any reporter.

  This is the practical Phoenix integration: MLServe never depends on Phoenix, but everything a
  Phoenix application wants to *see* about inference arrives through `:telemetry`, and this
  module turns those events into metric definitions.

  Add them to your existing telemetry supervisor:

      defmodule MyAppWeb.Telemetry do
        use Supervisor
        import Telemetry.Metrics

        def metrics do
          [
            # ... your Phoenix, Ecto and VM metrics ...
          ] ++ MLServe.Telemetry.Metrics.metrics()
        end
      end

  ## Optional dependency

  `:telemetry_metrics` is an *optional* dependency. When it is not present this module returns an
  empty list rather than failing to compile, so MLServe stays at one runtime dependency for
  everyone who does not need metrics. Add it yourself to switch these on:

      {:telemetry_metrics, "~> 1.0"}

  ## Tags

  Every metric is tagged with `:model` and, where meaningful, `:version` — so a canary rollout
  shows up as two series you can compare directly, which is the whole point of
  `MLServe.canary/3`.
  """

  @doc """
  Returns the metric definitions, or `[]` when `:telemetry_metrics` is not available.

  ## Examples

      MLServe.Telemetry.Metrics.metrics()
      |> Enum.map(& &1.name)
      #=> [[:ml_serve, :prediction, :duration], ...]
  """
  @spec metrics() :: list()
  def metrics do
    if Code.ensure_loaded?(Telemetry.Metrics) do
      build()
    else
      []
    end
  end

  if Code.ensure_loaded?(Telemetry.Metrics) do
    defp build do
      import Telemetry.Metrics

      model_tags = [:model, :version, :backend]

      [
        summary("ml_serve.prediction.duration",
          event_name: [:ml_serve, :prediction, :stop],
          measurement: :duration,
          unit: {:native, :millisecond},
          tags: model_tags ++ [:result, :canary?],
          description: "End-to-end prediction latency, including queueing, hooks and cache"
        ),
        summary("ml_serve.prediction.inference_duration",
          event_name: [:ml_serve, :prediction, :stop],
          measurement: :inference_duration,
          unit: {:native, :millisecond},
          tags: model_tags,
          description: "Time spent inside the backend, excluding queueing"
        ),
        summary("ml_serve.prediction.queue_duration",
          event_name: [:ml_serve, :prediction, :stop],
          measurement: :queue_duration,
          unit: {:native, :millisecond},
          tags: model_tags,
          description: "Time a request waited before a worker picked it up"
        ),
        counter("ml_serve.prediction.count",
          event_name: [:ml_serve, :prediction, :stop],
          measurement: :duration,
          tags: model_tags ++ [:result, :canary?],
          description: "Predictions served, split by result and canary status"
        ),
        counter("ml_serve.prediction.exception.count",
          event_name: [:ml_serve, :prediction, :exception],
          measurement: :duration,
          tags: model_tags,
          description: "Predictions where the backend raised"
        ),
        distribution("ml_serve.prediction.batch_size",
          event_name: [:ml_serve, :prediction, :stop],
          measurement: :batch_size,
          tags: [:model, :version],
          reporter_options: [buckets: [1, 2, 4, 8, 16, 32, 64, 128]],
          description: "Inputs per prediction call"
        ),
        summary("ml_serve.model.load.duration",
          event_name: [:ml_serve, :model, :load],
          measurement: :duration,
          unit: {:native, :millisecond},
          tags: model_tags ++ [:result],
          description: "How long a model took to load"
        ),
        counter("ml_serve.model.unload.count",
          event_name: [:ml_serve, :model, :unload],
          measurement: :duration,
          tags: model_tags,
          description: "Model unloads"
        ),
        summary("ml_serve.model.unload.drained",
          event_name: [:ml_serve, :model, :unload],
          measurement: :drained,
          tags: [:model, :version],
          description: "Requests still in flight when the drain timeout expired; 0 is clean"
        ),
        counter("ml_serve.cache.hit.count",
          event_name: [:ml_serve, :cache, :hit],
          measurement: :count,
          tags: [:model, :version],
          description: "Inference cache hits"
        ),
        counter("ml_serve.cache.miss.count",
          event_name: [:ml_serve, :cache, :miss],
          measurement: :count,
          tags: [:model, :version],
          description: "Inference cache misses"
        ),
        summary("ml_serve.batch.flush.size",
          event_name: [:ml_serve, :batch, :flush],
          measurement: :size,
          tags: [:model, :version, :reason],
          description: "Dynamic batch size at flush; mostly :timeout means the window is too long"
        ),
        summary("ml_serve.batch.flush.wait_duration",
          event_name: [:ml_serve, :batch, :flush],
          measurement: :wait_duration,
          unit: {:native, :millisecond},
          tags: [:model, :version],
          description: "How long the first input in a batch waited before the flush"
        )
      ]
    end
  else
    defp build, do: []
  end
end
