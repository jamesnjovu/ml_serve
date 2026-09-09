# Model Versioning

Models change more often than the applications that serve them, and a retrained model should not
require a deploy. MLServe keys models by `{name, version}` so several versions run side by side,
and routing between them is a pointer you move.

## Versions are independent models

```elixir
MLServe.load_model(:fraud_detection, backend: MyApp.Backends.ONNX, path: "fraud-v1.onnx", version: "1.0.0")
MLServe.load_model(:fraud_detection, backend: MyApp.Backends.ONNX, path: "fraud-v2.onnx", version: "2.1.0")
```

Each gets its own supervision subtree, its own workers, its own counters and its own telemetry
tags. They share nothing but a name.

```elixir
MLServe.versions(:fraud_detection)
#=> {:ok, ["1.0.0", "2.1.0"]}

MLServe.predict(:fraud_detection, features, version: "2.1.0")   # pinned
MLServe.predict(:fraud_detection, features)                     # default version
```

The first version registered under a name becomes the default. Loading another **does not** move
it — promotion is always explicit.

## The rollout

```elixir
# 1. Load the candidate beside the live model. Nothing changes for callers yet.
{:ok, _} = MLServe.load_model(:fraud_detection,
  backend: MyApp.Backends.ONNX,
  path: "fraud-v2.onnx",
  version: "2.1.0",
  workers: 4
)

:ok = MLServe.await_ready({:fraud_detection, "2.1.0"})

# 2. Send it a slice of traffic.
:ok = MLServe.canary(:fraud_detection, "2.1.0", 5)

# 3. Watch. Telemetry is already tagged by version.

# 4. Promote — an atomic pointer flip. No restart, no dropped request.
:ok = MLServe.promote(:fraud_detection, "2.1.0")

# 5. Drain and remove the old version.
:ok = MLServe.unload_model(:fraud_detection, version: "1.0.0")
```

At no point does the application restart, and at no point does a request fail because of the
change. MLServe's own test suite asserts exactly this: 600 concurrent predictions across a full
load → canary → promote → drain cycle, with zero failures.

## Canary routing

```elixir
MLServe.canary(:fraud_detection, "2.1.0", 5)     # 5% of unpinned traffic
MLServe.clear_canary(:fraud_detection)           # abort
```

Each request rolls independently **in the calling process** against its own random seed. There is
no shared counter and no coordination point, so the split costs nothing and cannot become a
bottleneck.

Rules worth knowing:

- **Pinned requests ignore the canary.** `version: "1.0.0"` goes exactly there.
- **A candidate that is not ready never black-holes its share.** If it is still loading or has
  failed, those requests fall back to the default version.
- **`promote/2` clears the canary.** Promoting is the end of the rollout, not another state to
  clean up.
- **Unloading the candidate clears the canary** too.

## Deciding with telemetry

Every prediction event carries `version` and `canary?`, so per-version comparison needs no extra
instrumentation. That is the whole reason the canary exists — a rollout you cannot measure is just
a slower way to ship a bad model.

```elixir
:telemetry.attach_many("canary-watch", [[:ml_serve, :prediction, :stop]], fn _e, m, meta, _c ->
  :telemetry.execute(
    [:my_app, :inference],
    %{duration: m.duration},
    %{model: meta.model, version: meta.version, canary: meta.canary?, result: meta.result}
  )
end, nil)
```

With `MLServe.Telemetry.Metrics.metrics/0`, latency and error-rate series are already split by
`version`. Compare, then promote or clear.

What to look at before promoting:

| | |
| --- | --- |
| Error rate | Higher on the candidate → clear the canary |
| `inference_duration` p99 | A slower model may still be fine, but budget for it |
| `prediction.exception` count | Any at all means a backend bug, not a model regression |
| Prediction distribution | Your own metric — a model that never predicts `:fraud` is broken in a way latency will not show |

## Graceful drain

`unload_model/2` does not kill in-flight work:

1. The version is marked `:draining`, so the registry stops routing new requests to it.
2. MLServe waits for the in-flight atomic counter to reach zero, bounded by `:drain_timeout`.
3. Workers terminate; the backend's `unload/1` runs; cached entries for that version are dropped.
4. `[:ml_serve, :model, :unload]` reports `drained` — the number still outstanding when the wait
   ended. `0` is a clean drain.

```elixir
models: [fraud_detection: [drain_timeout: 15_000]]
```

Set `:drain_timeout` above your p99 inference latency. Alert on `drained > 0`: it means a deploy
abandoned live requests.

If you unload the version that is currently the default and others remain, the newest survivor
becomes the default automatically, so an unpinned `predict/3` keeps working rather than starting
to return `:model_not_found`.

## Version strings

Any non-empty string works. `MLServe.versions/1` sorts dot- and dash-separated integer components
numerically, so `"10.0.0"` correctly follows `"9.0.0"` rather than sorting before it. Anything
non-numeric falls back to string ordering.

Semver is the obvious choice, but a training run id or a date works equally well:

```elixir
version: "2026-08-22-run-4417"
```

Whatever you choose, make it the thing you would want to see in an alert at 3am.

## Reload versus a new version

```elixir
MLServe.reload_model(:fraud_detection)                  # same version, re-read the artifact
MLServe.reload_model(:fraud_detection, workers: 8)      # with overrides
```

`reload_model/2` is `unload` then `load` on the same version, which means **it is a gap in
availability**. It is right for picking up a changed config or a hot-swapped file in development.

For anything user-facing, load a new version and promote. That path has no gap at all.

## Loading versions at boot

```elixir
config :ml_serve,
  models: [
    fraud_detection: [
      backend: MyApp.Backends.ONNX,
      path: "fraud-v2.onnx",
      version: "2.1.0",
      checksum: {:sha256, System.get_env("FRAUD_MODEL_SHA256")},
      workers: 4
    ]
  ]
```

Pinning the version and checksum in config makes deploys reproducible: the same release always
loads the same artifact, and a mismatch fails loudly at load time rather than serving predictions
from a file you did not intend to ship.

## Registering versions from a database

Runtime registration goes through exactly the same validation as configuration, so a model
registry table works naturally:

```elixir
defmodule MyApp.ModelLoader do
  use Task, restart: :transient

  def start_link(_), do: Task.start_link(__MODULE__, :run, [])

  def run do
    for model <- MyApp.Repo.all(MyApp.ModelVersion.active()) do
      MLServe.load_model(model.name,
        backend: MyApp.Backends.ONNX,
        path: model.path,
        version: model.version,
        checksum: {:sha256, model.sha256},
        workers: model.workers
      )
    end

    for model <- MyApp.Repo.all(MyApp.ModelVersion.canaries()) do
      MLServe.canary(model.name, model.version, model.traffic_percent)
    end
  end
end
```

Add it to your own supervision tree after MLServe has started. Because loading is asynchronous and
retried with backoff, a temporarily unreachable artifact store delays a model rather than taking
your application down.
