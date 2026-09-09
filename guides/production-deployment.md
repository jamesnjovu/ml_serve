# Production Deployment

Getting MLServe into production: where model files live, how to size a node, what to alert on, and
how to deploy without dropping requests.

## Shipping model artifacts

Three options, with different trade-offs:

| Approach | Pro | Con |
| --- | --- | --- |
| **In the release** (`priv/models/`) | Immutable, atomic with the code, no runtime fetch | Bigger images; a model change needs a deploy |
| **Object storage, fetched at boot** | Decoupled from deploys | Boot depends on the network; needs a fetch step |
| **Mounted volume** | No fetch, shared across instances | Attaches asynchronously; the mount may not be ready at boot |

MLServe's asynchronous loading with backoff retry is what makes the latter two safe: a model on a
volume that attaches three seconds after the container starts loads on the second attempt instead
of taking the application down.

### In the release

```elixir
# mix.exs
defp releases do
  [my_app: [include_executables_for: [:unix], applications: [runtime_tools: :permanent]]]
end
```

`priv/` is included in a release automatically. Point `:model_root` at it:

```elixir
# config/runtime.exs
config :ml_serve,
  model_root: Path.join(:code.priv_dir(:my_app), "models"),
  models: [
    fraud_detection: [
      backend: MyApp.Backends.ONNX,
      path: "fraud-v2.onnx",
      version: System.get_env("FRAUD_MODEL_VERSION", "2.1.0"),
      checksum: {:sha256, System.fetch_env!("FRAUD_MODEL_SHA256")},
      workers: String.to_integer(System.get_env("FRAUD_WORKERS", "4"))
    ]
  ]
```

`:code.priv_dir/1` rather than a relative path: the working directory of a running release is not
where you think it is.

### Fetching at boot

```elixir
defmodule MyApp.ModelFetcher do
  use Task, restart: :transient

  def start_link(_), do: Task.start_link(__MODULE__, :run, [])

  def run do
    dest = Path.join(MLServe.Config.model_root(), "fraud-v2.onnx")

    unless File.exists?(dest) do
      File.mkdir_p!(Path.dirname(dest))
      MyApp.Storage.download!("models/fraud-v2.onnx", dest)
    end

    MLServe.load_model(:fraud_detection,
      backend: MyApp.Backends.ONNX,
      path: "fraud-v2.onnx",
      version: "2.1.0",
      checksum: {:sha256, System.fetch_env!("FRAUD_MODEL_SHA256")}
    )
  end
end
```

Start it in your own supervision tree, after MLServe's application. Keep it out of
`Application.start/2` so a fetch failure does not prevent the rest of your app — including the
health endpoint that would tell you why — from booting.

## Security

```elixir
config :ml_serve,
  model_root: Path.join(:code.priv_dir(:my_app), "models"),
  max_model_bytes: 4 * 1024 * 1024 * 1024
```

MLServe validates every configured path before the backend sees it:

- resolves inside `:model_root`, with `..` traversal **and symlink escape** rejected — symlinks are
  resolved before the containment check, so a link inside the root pointing at `/etc` does not pass
- exists, is a regular file, is readable
- is no larger than `:max_model_bytes`
- matches `:checksum` when one is configured

Checksums are worth setting even when the artifact ships in the release. They turn "we deployed
the wrong model six hours ago" into a loud failure at load time.

> #### Model files are data {: .info}
>
> MLServe never calls `binary_to_term/1`, `Code.eval_*`, or loads a NIF from a configured path.
> Supplying a model file is not a way to execute code. If **your** backend deserialises an
> artifact, use `:erlang.binary_to_term(bin, [:safe])`.

Never take `:path` or `:backend` from user input. `:backend` is a module that gets called; `:path`
is constrained to `:model_root` but there is no reason to let a request choose a file at all.

## Sizing

Start from the backend, not from the machine:

| Backend | `workers` | Notes |
| --- | --- | --- |
| `Nx.Serving` / Bumblebee | n/a — use `:shared` | Nx does its own batching; a pool would only add copies |
| ONNX Runtime, CPU | 1–2 | ORT is already multi-threaded internally |
| ONNX Runtime, GPU | 1–2 + `batching:` | One device; the batcher is what keeps it fed |
| Python port | one per core, minus headroom | Each is an OS process |
| Remote HTTP | `:shared` + `max_concurrency:` | No pool needed |

Memory: with `load: :once` (the default) a NIF-resource model is loaded once regardless of pool
size. With `:per_worker`, multiply by `workers` and check it fits — this is the single most common
way to OOM a node with MLServe.

Set an admission limit so overload sheds instead of queueing:

```elixir
models: [fraud_detection: [workers: 4, max_concurrency: 64, timeout: 2_000]]
```

## Zero-downtime deploys

Two separate concerns.

### Deploying new code

Standard rolling deploy. New instances load models on boot and report `ready?() == false` until
they can serve, so a readiness probe keeps traffic away until then:

```yaml
readinessProbe:
  httpGet: { path: /health/ready, port: 4000 }
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 12          # allow a slow model load

livenessProbe:
  httpGet: { path: /health/live, port: 4000 }
  periodSeconds: 10
```

Keep liveness independent of model state. Tying them together turns one unhealthy model into an
endless kill-restart loop across the fleet.

Set `initialDelaySeconds` and `failureThreshold` to cover your slowest model load — check
`load_duration_ms` in `MLServe.model_status/2`.

### Deploying a new model

No deploy at all — see [Model Versioning](model-versioning.md):

```elixir
MLServe.load_model(:fraud_detection, version: "2.2.0", backend: ..., path: ...)
MLServe.await_ready({:fraud_detection, "2.2.0"})
MLServe.canary(:fraud_detection, "2.2.0", 5)
# watch per-version telemetry
MLServe.promote(:fraud_detection, "2.2.0")
MLServe.unload_model(:fraud_detection, version: "2.1.0")
```

## Graceful shutdown

Set the shutdown timeout above your p99 inference latency so in-flight predictions finish:

```elixir
# config/runtime.exs
config :my_app, MyAppWeb.Endpoint, drainer: [shutdown: 30_000]
```

```yaml
terminationGracePeriodSeconds: 45     # comfortably above the drainer
```

`MLServe.unload_model/2` drains on its own, and `[:ml_serve, :model, :unload]` reports `drained` —
the number still in flight when the wait ended. **Alert on `drained > 0`**: it means a deploy
abandoned live requests.

```elixir
models: [fraud_detection: [drain_timeout: 15_000]]
```

## Observability

Wire up metrics before you need them:

```elixir
def metrics do
  [...your metrics...] ++ MLServe.Telemetry.Metrics.metrics()
end
```

What to alert on:

| Condition | Meaning | Action |
| --- | --- | --- |
| `model.load` with `result: :error` | An instance is running without a model | Page. Check the path and checksum. |
| `ready?()` false past the probe window | Model never loaded | Page. |
| `prediction` error rate up on one `version` | Bad model | Clear the canary or roll back. |
| `prediction.exception` > 0 | Backend bug | Stacktrace is in the metadata. |
| `queue_duration` p99 rising, `inference_duration` flat | Pool too small | Add workers. |
| `inference_duration` p99 rising | Model or machine slower | Investigate the node. |
| `:overloaded` rate climbing | Hitting `max_concurrency` | Scale out, or raise the limit if there is headroom. |
| `model.unload` with `drained > 0` | Deploy dropped requests | Raise `:drain_timeout`. |

In production, log model lifecycle only and send the rest to metrics:

```elixir
MLServe.Telemetry.Logger.attach(level: :info, events: [:model])
```

Logging every prediction spends your I/O budget on strings.

## Multi-node

MLServe is **node-local by design**: each node loads its own models and has its own catalog,
counters and cache. There is no distributed coordination, and none is needed — inference is
stateless and your load balancer already spreads requests.

Two consequences worth planning for:

- **`load_model/2`, `promote/2` and `canary/3` are per-node.** Driving a rollout across a cluster
  means calling them on every node. Either loop over `Node.list/0` with `:erpc`, or — better —
  drive it from configuration and let each node converge on boot:

  ```elixir
  for node <- [Node.self() | Node.list()] do
    :erpc.call(node, MLServe, :promote, [:fraud_detection, "2.2.0"])
  end
  ```

- **A canary percentage is per-node but converges globally.** Each node rolls independently, so
  5% on every node is 5% overall.

Keeping a model registry table in Postgres and having each node load from it on boot is usually
cleaner than orchestrating RPCs — see the end of the
[Model Versioning](model-versioning.md) guide.

## Cache in production

Off by default. Turn it on only where the same input genuinely must produce the same output —
embeddings of immutable documents, scores for a stable feature vector:

```elixir
config :ml_serve,
  cache: [enabled: true, max_size: 50_000, ttl: :timer.minutes(5), sweep_interval: :timer.minutes(1)]
```

The cache is per node and in memory. `:max_size` bounds it, but the real bound is entry size:
50 000 embeddings of 1 536 floats is roughly 300 MB. Size it against the memory you actually have.

Watch the hit rate. Below about 20%, the cache is costing you hashing time and memory to avoid
very little work — turn it off, or use a cheaper `:cache_key`.

## Checklist

- [ ] `:model_root` set to an absolute path via `:code.priv_dir/1`
- [ ] `:checksum` set for every model
- [ ] `workers` sized for the backend, not copied from a blog post
- [ ] `max_concurrency` and `timeout` set so overload sheds rather than queues
- [ ] `/health/ready` uses `MLServe.ready?/0`; `/health/live` does not
- [ ] Probe `failureThreshold` covers the slowest `load_duration_ms`
- [ ] `drain_timeout` above p99 inference latency; alerting on `drained > 0`
- [ ] `MLServe.Telemetry.Metrics.metrics/0` wired into your reporter
- [ ] Alerts on failed loads and per-version error rate
- [ ] A rollback plan: the previous version stays loaded until the new one is proven
