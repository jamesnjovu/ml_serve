# Telemetry — seeing what inference is actually doing.
#
#     elixir examples/scripts/06_telemetry.exs
#
# Covers: the built-in logger, the events MLServe emits, turning the prediction span into
# latency percentiles, reading queue time against inference time to tell a slow model from a
# small pool, and wiring Telemetry.Metrics up for LiveDashboard.

Mix.install([
  # Running from a clone of the repo. Against the published package this is:
  #   {:ml_serve, "~> 0.1.0"}
  {:ml_serve, path: Path.expand("../..", __DIR__)},
  # Optional. MLServe.Telemetry.Metrics returns [] when this is absent, so MLServe stays at one
  # runtime dependency for everyone who does not need metric definitions.
  {:telemetry_metrics, "~> 1.0"}
])

section = fn title ->
  IO.puts("\n── " <> title <> " " <> String.duplicate("─", max(0, 62 - String.length(title))))
end

defmodule VariableModel do
  @moduledoc "Inference time varies with the input, so the percentiles below mean something."
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :exclusive, load: :once}

  @impl true
  def load(_config), do: {:ok, nil}

  @impl true
  def predict(_state, :slow), do: Process.sleep(40) && {:ok, :slow}
  def predict(_state, :raises), do: raise(ArgumentError, "the weights are not finite")
  def predict(_state, :rejects), do: {:error, :input_out_of_domain}
  def predict(_state, _input), do: Process.sleep(2) && {:ok, :fast}
end

defmodule Latency do
  @moduledoc """
  Turns the prediction span into per-model percentiles.

  A real deployment sends these to StatsD, Prometheus or LiveDashboard rather than an ETS bag,
  but the handler is the same shape: read the measurements, tag with the metadata.
  """

  def start do
    :ets.new(:latency, [:public, :named_table, :duplicate_bag])

    :telemetry.attach(
      "latency",
      [:ml_serve, :prediction, :stop],
      &__MODULE__.handle_event/4,
      nil
    )
  end

  def handle_event(_event, measurements, metadata, _config) do
    :ets.insert(:latency, {
      {metadata.model, metadata.result},
      us(measurements.duration),
      us(measurements.queue_duration),
      us(measurements.inference_duration)
    })

    :ok
  end

  defp us(native), do: System.convert_time_unit(native, :native, :microsecond)

  def reset, do: :ets.delete_all_objects(:latency)

  def count(model, result \\ :ok), do: length(:ets.lookup(:latency, {model, result}))

  def percentiles(model) do
    durations =
      :latency
      |> :ets.lookup({model, :ok})
      |> Enum.map(fn {_key, duration, _q, _i} -> duration end)
      |> Enum.sort()

    at = fn p -> Enum.at(durations, min(round(length(durations) * p), length(durations) - 1)) end

    %{count: length(durations), p50: at.(0.5), p95: at.(0.95), p99: at.(0.99)}
  end

  def split(model) do
    rows = :ets.lookup(:latency, {model, :ok})
    avg = fn f -> div(Enum.sum(Enum.map(rows, f)), max(length(rows), 1)) end

    %{
      total: avg.(fn {_k, d, _q, _i} -> d end),
      queue: avg.(fn {_k, _d, q, _i} -> q end),
      inference: avg.(fn {_k, _d, _q, i} -> i end)
    }
  end
end

load = fn requests, concurrency ->
  1..requests
  |> Task.async_stream(
    fn i -> MLServe.predict(:variable, if(rem(i, 10) == 0, do: :slow, else: :fast)) end,
    max_concurrency: concurrency,
    timeout: 60_000
  )
  |> Stream.run()
end

ms = fn us -> Float.round(us / 1_000, 1) end

section.("Every event MLServe emits")

for event <- MLServe.Telemetry.events() do
  IO.puts("    #{inspect(event)}")
end

section.("The built-in logger")

# One line for visibility with no handler of your own. :events accepts the :prediction, :model,
# :cache and :batch groups; :level defaults to :debug.
MLServe.Telemetry.Logger.attach(level: :info, events: [:model])

{:ok, _} = MLServe.load_model(:variable, backend: VariableModel, workers: 4)
:ok = MLServe.await_ready(:variable)

MLServe.Telemetry.Logger.detach()

section.("Latency percentiles from the prediction span")

Latency.start()

# Four workers, four concurrent callers: nobody waits, so these numbers are the model itself.
load.(300, 4)

stats = Latency.percentiles(:variable)

IO.puts("""
    requests   #{stats.count}
    p50        #{ms.(stats.p50)}ms
    p95        #{ms.(stats.p95)}ms
    p99        #{ms.(stats.p99)}ms

    One request in ten is deliberately 20× slower. The mean would have buried it; p99 is where
    it shows up, which is the entire argument for exporting a distribution rather than an
    average.\
""")

section.("Queue time vs inference time")

# :duration is end to end — admission, hooks, cache and all. :queue_duration is the wait before
# a worker picked the request up, and :inference_duration is the time inside the backend
# callback. Keeping them apart is what separates "the model got slower" from "the pool is too
# small", which have completely different fixes.
before = Latency.split(:variable)

Latency.reset()
load.(300, 40)
oversubscribed = Latency.split(:variable)

IO.puts("""
                        4 callers    40 callers
    total             #{String.pad_leading(to_string(ms.(before.total)), 8)}ms  #{String.pad_leading(to_string(ms.(oversubscribed.total)), 10)}ms
    queued            #{String.pad_leading(to_string(ms.(before.queue)), 8)}ms  #{String.pad_leading(to_string(ms.(oversubscribed.queue)), 10)}ms
    inference         #{String.pad_leading(to_string(ms.(before.inference)), 8)}ms  #{String.pad_leading(to_string(ms.(oversubscribed.inference)), 10)}ms

    Inference time barely moved; queue time exploded. The model is fine — the pool is too small.
    Both measurements are 0 for :shared backends, which never queue: inference runs in the
    caller.\
""")

section.("How failures show up")

Latency.reset()

# A backend that returns {:error, reason}.
{:error, rejected} = MLServe.predict(:variable, :rejects)

# A backend that *raises*. MLServe catches it at the backend boundary, preserves the exception
# and stacktrace in a MLServe.BackendError, and keeps serving. The worker is not restarted —
# set restart_on_error: true if you would rather it were.
{:error, {:backend_error, error}} = MLServe.predict(:variable, :raises)

IO.puts("""
    returned error    #{inspect(rejected)}
    raised            #{inspect(error.reason)}
      backend         #{inspect(error.backend)}
      callback        #{inspect(error.callback)}
      retryable?      #{MLServe.Error.retryable?(error)}

    Both arrive as a :stop event with result: :error — MLServe catches the raise before it can
    escape, so from telemetry's point of view a crash and a rejection look the same. Tell them
    apart by the return value: only a raise carries a MLServe.BackendError.

    :stop events recorded: #{Latency.count(:variable, :error)} with result: :error\
""")

section.("Telemetry.Metrics for LiveDashboard")

# Append these to your existing MyAppWeb.Telemetry metrics/0 list and inference shows up beside
# Phoenix and Ecto. Every metric is tagged with :model and, where meaningful, :version — so a
# canary rollout renders as two directly comparable series.
metrics = MLServe.Telemetry.Metrics.metrics()

IO.puts("    #{length(metrics)} metric definitions:\n")

for metric <- metrics do
  kind = metric.__struct__ |> Module.split() |> List.last()
  IO.puts("    #{String.pad_trailing(kind, 14)} #{Enum.join(metric.name, ".")}")
end

MLServe.unload_model(:variable)
