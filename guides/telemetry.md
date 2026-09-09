# Telemetry

MLServe treats instrumentation as a feature. `:telemetry` is its only runtime dependency, every
prediction is wrapped in a span, and an unattached event costs a single ETS lookup.

## Events

```text
[:ml_serve, :prediction, :start]
[:ml_serve, :prediction, :stop]
[:ml_serve, :prediction, :exception]
[:ml_serve, :model, :load]
[:ml_serve, :model, :unload]
[:ml_serve, :cache, :hit]
[:ml_serve, :cache, :miss]
[:ml_serve, :batch, :flush]
```

`MLServe.Telemetry.events/0` returns the list, which is convenient for
`:telemetry.attach_many/4`.

## The prediction span

### Measurements on `:stop`

| | |
| --- | --- |
| `duration` | End-to-end: routing, cache, hooks, dispatch, inference |
| `queue_duration` | How long the request waited before a worker picked it up |
| `inference_duration` | Time inside the backend callback |
| `batch_size` | `1` for `predict/3`, the list length for `batch_predict/3` |

Separating `queue_duration` from `inference_duration` is what lets you answer the only question
that matters when latency rises: *is the model slower, or is the pool too small?* A climbing
`queue_duration` with flat `inference_duration` means add workers. The reverse means the model or
the machine changed.

`queue_duration` and `inference_duration` are `0` for `:shared` backends, which never queue.

### Metadata

| | |
| --- | --- |
| `model` | Model name |
| `version` | The version that actually served the request |
| `backend` | Backend module |
| `batch?` | Came from `batch_predict/3` |
| `canary?` | Canary routing chose this version |
| `cached?` | Result came from the cache (`:stop` only) |
| `result` | `:ok` or `:error` (`:stop` only) |

Because `version` and `canary?` ride on every event, comparing a canary against the incumbent
needs no extra instrumentation. That is what makes a promote-or-roll-back decision possible — see
[Model Versioning](model-versioning.md).

### `:exception` versus an error result

- A backend that **raises** produces `[:ml_serve, :prediction, :exception]` with `kind`, `reason`
  and `stacktrace`.
- A backend that **returns** `{:error, reason}` produces a normal `:stop` event with
  `result: :error`.

The distinction is deliberate. An expected rejection is not an exception, and conflating them
makes your error-rate dashboard useless: you cannot tell "the model declined this input" from "the
model is broken".

## Lifecycle events

`[:ml_serve, :model, :load]` — measurements `duration`; metadata `model`, `version`, `backend`,
`workers`, `result`.

`[:ml_serve, :model, :unload]` — measurements `duration` and `drained` (requests still in flight
when the drain timeout expired; `0` is a clean drain); metadata `model`, `version`, `backend`.

These are single events with a duration rather than span triples, because a load either happened
or it did not — there is no useful window to observe in between.

`drained` is worth alerting on: a non-zero value means a deploy abandoned live requests, and
either your `:drain_timeout` is too short or something is wedged.

## Cache and batching

`[:ml_serve, :cache, :hit]` / `[:ml_serve, :cache, :miss]` — measurements `count: 1`; metadata
`model`, `version`.

`[:ml_serve, :batch, :flush]` — measurements `size` and `wait_duration`; metadata `model`,
`version`, `reason` (`:full` or `:timeout`). A healthy dynamic-batching setup flushes mostly on
`:full`; mostly `:timeout` means the window is longer than your traffic warrants.

## The quick way: log it

```elixir
MLServe.Telemetry.Logger.attach(level: :info)
```

```text
[ml_serve] loaded :fraud_detection v1.0.0 (ok) in 812.4ms with 4 worker(s)
[ml_serve] :fraud_detection v1.0.0 ok in 4.21ms (inference 3.9ms, queue 0.11ms, batch 1)
[ml_serve] :fraud_detection v2.1.0 ok in 3.88ms (inference 3.6ms, queue 0.08ms, batch 1, canary)
[ml_serve] unloaded :fraud_detection v1.0.0 in 41.2ms, drained cleanly
```

Options: `:level` (default `:debug`) and `:events` — one or more of `:prediction`, `:model`,
`:cache`, `:batch`, defaulting to `[:prediction, :model]`.

Logging every prediction in production spends your I/O budget on strings. In production, attach
`events: [:model]` and send the rest to a metrics backend.

## The production way: metrics

```elixir
# lib/my_app_web/telemetry.ex
defmodule MyAppWeb.Telemetry do
  use Supervisor
  import Telemetry.Metrics

  def start_link(arg), do: Supervisor.start_link(__MODULE__, arg, name: __MODULE__)

  @impl true
  def init(_arg) do
    children = [
      {:telemetry_poller, measurements: periodic_measurements(), period: 10_000}
      # {Telemetry.Metrics.ConsoleReporter, metrics: metrics()}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end

  def metrics do
    [
      # ... your Phoenix, Ecto and VM metrics ...
    ] ++ MLServe.Telemetry.Metrics.metrics()
  end

  defp periodic_measurements, do: []
end
```

`MLServe.Telemetry.Metrics.metrics/0` returns definitions for latency, throughput, errors, cache
hit rate, load duration, drain counts and batch sizes — every one tagged with `:model` and, where
meaningful, `:version`.

`:telemetry_metrics` is an **optional** dependency. Without it the function returns `[]` rather
than failing to compile, so MLServe stays at one runtime dependency for everyone who does not need
metrics. Add it to switch them on:

```elixir
{:telemetry_metrics, "~> 1.0"}
```

### LiveDashboard

```elixir
live_dashboard "/dashboard", metrics: MyAppWeb.Telemetry
```

Model latency, throughput and cache hit rate appear on the Metrics tab beside your Phoenix and
Ecto charts.

## Custom handlers

```elixir
:telemetry.attach_many(
  "ml-serve-slow-predictions",
  [[:ml_serve, :prediction, :stop]],
  fn _event, measurements, metadata, _config ->
    ms = System.convert_time_unit(measurements.duration, :native, :millisecond)

    if ms > 500 do
      Logger.warning("[slow] #{metadata.model} v#{metadata.version} took #{ms}ms")
    end
  end,
  nil
)
```

Alerting on a failed load:

```elixir
:telemetry.attach("ml-serve-load-failed", [:ml_serve, :model, :load], fn _e, _m, meta, _c ->
  if meta.result == :error do
    MyApp.Alerts.page("model #{meta.model} v#{meta.version} failed to load")
  end
end, nil)
```

> #### Handlers run inline {: .warning}
>
> A telemetry handler executes in the process that emitted the event — for a prediction, that is
> the caller. Keep handlers cheap and non-blocking. Anything involving I/O belongs in a separate
> process; send it a message rather than doing the work in the handler.

## Testing telemetry

`:telemetry_test` ships with `:telemetry`:

```elixir
test "emits a stop event" do
  ref = :telemetry_test.attach_event_handlers(self(), [[:ml_serve, :prediction, :stop]])

  MLServe.predict(:fraud_detection, features)

  assert_receive {[:ml_serve, :prediction, :stop], ^ref, measurements, %{model: :fraud_detection}}
  assert measurements.duration > 0
end
```

> #### Match on the model name {: .warning}
>
> Telemetry handlers are **global**. In an `async: true` suite your handler fires for events from
> *other tests' models too*, and the `ref` does not isolate them — it identifies the handler, not
> the emitter. Always pattern-match `%{model: ^name}`, or you will write a test that passes alone
> and fails in a full run.

## Suggested alerts

| Condition | Meaning |
| --- | --- |
| `result: :error` rate rising for one `version` | A bad model — roll back the canary |
| `prediction.exception` count > 0 | A backend bug; the stacktrace is in the metadata |
| `queue_duration` p99 rising, `inference_duration` flat | Pool too small |
| `inference_duration` p99 rising | Model or machine got slower |
| `model.unload` with `drained > 0` | A deploy abandoned live requests |
| `model.load` with `result: :error` | An instance is running without a model |
| `batch.flush` mostly `:timeout` | Batch window longer than traffic justifies |
