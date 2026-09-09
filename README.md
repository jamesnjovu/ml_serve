<h1 align="center">MLServe</h1>

<p align="center">
  <strong>Production machine-learning inference for the BEAM.</strong><br>
  OTP supervision, worker pools, dynamic batching, caching, telemetry and zero-downtime model
  rollout — around any ML backend.
</p>

<p align="center">
  <a href="https://hex.pm/packages/ml_serve"><img src="https://img.shields.io/hexpm/v/ml_serve.svg?style=flat-square" alt="Hex version"></a>
  <a href="https://hexdocs.pm/ml_serve"><img src="https://img.shields.io/badge/hex-docs-blue.svg?style=flat-square" alt="HexDocs"></a>
  <a href="https://hex.pm/packages/ml_serve"><img src="https://img.shields.io/hexpm/dt/ml_serve.svg?style=flat-square" alt="Downloads"></a>
  <a href="https://github.com/jamesnjovu/ml_serve/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/jamesnjovu/ml_serve/ci.yml?branch=main&style=flat-square" alt="CI"></a>
  <a href="https://github.com/jamesnjovu/ml_serve/blob/main/LICENSE"><img src="https://img.shields.io/hexpm/l/ml_serve.svg?style=flat-square" alt="MIT"></a>
</p>

---

Elixir has excellent *model* libraries — Nx, Bumblebee, Ortex, EXLA. What it has not had is the
boring layer around them: the thing that supervises a model, pools access to it, batches requests,
caches results, reports latency, and lets you swap a model version without a deploy.

Every team that puts a model into a Phoenix app rebuilds that layer by hand. MLServe *is* that
layer, and nothing else. It does not train, it does not own a tensor type, and it is not another
LLM API wrapper.

```elixir
MLServe.predict(:fraud_detection, %{
  amount: 1500.50,
  transaction_count_24h: 8,
  failed_transactions_24h: 2
})
#=> {:ok, %{prediction: :fraud, probability: 0.94}}
```

## Installation

Add `ml_serve` to your dependencies in `mix.exs`:

```elixir
def deps do
  [{:ml_serve, "~> 0.1.0"}]
end
```

```bash
mix deps.get
```

Requires Elixir 1.14 or later. The only runtime dependency is `:telemetry`.

## Why it exists

Inference serving is a concurrency problem wearing a machine-learning hat: parallel requests,
stateful and expensive-to-load models, backends that are not thread-safe, failures that must stay
contained, and versions that must change without downtime. That list is OTP's home ground.

What that buys you concretely:

- **Per-model isolation.** Each loaded model version is its own supervision subtree. A backend that
  crash-loops exhausts its own restart budget and marks *itself* failed. Other models keep serving.
- **No dispatch overhead.** Finding a model is one lock-free ETS read in the calling process.
  MLServe never puts a GenServer between your request and the model.
- **Shared-state backends cost nothing.** A backend safe to call concurrently — an `Nx.Serving`,
  a pure function, a remote service — runs *in the caller* with state read from `:persistent_term`.
  No worker processes, and your tensors are never copied between mailboxes.
- **Dynamic batching.** Independent concurrent callers are coalesced into one backend call, which
  is the difference between a busy GPU and an idle one.
- **Zero-downtime model upgrades.** Load a new version beside the running one, send it 5% of
  traffic, compare per-version telemetry, promote, drain the old one.

## Quick start

Any module with `load/1` and `predict/2` is a model:

```elixir
defmodule MyApp.FraudModel do
  @behaviour MLServe.Model

  @impl true
  def load(config), do: {:ok, Keyword.fetch!(config, :threshold)}

  @impl true
  def predict(threshold, %{amount: amount}) do
    probability = min(amount / 2000, 1.0)

    {:ok, %{
      prediction: if(probability > threshold, do: :fraud, else: :legitimate),
      probability: probability
    }}
  end
end
```

Register it:

```elixir
config :ml_serve,
  models: [
    fraud_detection: [
      backend: MyApp.FraudModel,
      version: "1.0.0",
      workers: 4,
      config: [threshold: 0.7]
    ]
  ]
```

…or at runtime, which is the same code path:

```elixir
MLServe.load_model(:fraud_detection, backend: MyApp.FraudModel, config: [threshold: 0.7])
```

Then predict from anywhere — a controller, an Oban job, a `Task`:

```elixir
case MLServe.predict(:fraud_detection, %{amount: 1500.50}) do
  {:ok, result} -> result
  {:error, reason} -> Logger.warning("inference failed: #{inspect(reason)}")
end
```

## Architecture

```text
MLServe.Supervisor                     (:rest_for_one)
├── MLServe.Registry                   process registry, partitioned by scheduler
├── MLServe.ModelRegistry              owns the catalog ETS table
├── MLServe.Cache                      owns the cache ETS table + TTL sweeper
├── MLServe.TaskSupervisor             batch fan-out
└── MLServe.ModelSupervisor            DynamicSupervisor
      └── MLServe.ModelInstance        one subtree per loaded {name, version}
            ├── MLServe.ModelServer    lifecycle, status, graceful drain
            ├── MLServe.WorkerSupervisor
            │     └── MLServe.Worker × N
            └── MLServe.Batcher        when dynamic batching is configured
```

`:rest_for_one` at the top is deliberate: `MLServe.ModelRegistry` owns the catalog ETS table, and
ETS tables die with their owner. If it restarts, every route has evaporated and any model still
running would be serving traffic the registry no longer knows about.

The prediction path itself contains no MLServe process at all:

```text
route lookup (1 ETS read) → admission control (atomic counter) → telemetry span opens
  → cache lookup → preprocess hook → dispatch → postprocess hook → cache write
```

Cache, hooks and validation all run *before* dispatch, so a worker is occupied only for actual
inference. A `preprocess` hook that queries Postgres for stored features runs on the caller's own
scheduler time, never on a GPU worker's.

See the [Architecture guide](guides/architecture.md) for the full picture.

## Concurrency

A backend declares how it may be executed, and MLServe picks a completely different strategy:

| `concurrency` | Where `predict/2` runs | Where state lives | Use for |
| --- | --- | --- | --- |
| `:shared` | **The calling process** — no processes, no copies | `:persistent_term` | `Nx.Serving`, Bumblebee, pure functions, remote services |
| `:exclusive` | A pooled worker | Worker state, or a shared handle | ONNX Runtime sessions, Python ports, anything not thread-safe |

Crossed with when the model loads:

| `load` | Meaning |
| --- | --- |
| `:once` (default) | `load/1` runs once; the state term is shared by all workers. Correct for NIF-resource models, where the term is a cheap handle. |
| `:per_worker` | `load/1` runs per worker. Correct for ports and per-worker sessions. |

`:once` as the default matters — loading a 2 GB model separately into eight workers is an
out-of-memory crash, not a pool.

```elixir
models: [fraud_detection: [workers: 8, selection: :least_loaded, max_concurrency: 64]]
```

## Model lifecycle

```elixir
MLServe.load_model(:fraud, backend: MyApp.Model, version: "2.1.0")
MLServe.await_ready({:fraud, "2.1.0"})   # loading is async; boot is never blocked

MLServe.canary(:fraud, "2.1.0", 5)       # 5% of traffic, tagged in telemetry
MLServe.promote(:fraud, "2.1.0")         # atomic pointer flip, no restart
MLServe.unload_model(:fraud, version: "1.0.0")   # drains in-flight requests first

MLServe.model_status(:fraud)
#=> {:ok, %{status: :ready, workers: 4, in_flight: 3, requests: 154_223, errors: 12, ...}}
```

Loading is asynchronous and retried with backoff, so a model on a network mount that attaches a
moment after the container starts does not take your application down with it.
`MLServe.ready?/1` is your readiness probe.

## Batch inference

```elixir
MLServe.batch_predict(:fraud_detection, [features_a, features_b, features_c])
```

One backend call when the backend implements `batch_predict/2`, a mapped `predict/2` otherwise.

For *many concurrent callers* rather than one caller with many inputs, enable dynamic batching:

```elixir
models: [fraud_detection: [batching: [max_size: 16, timeout: 10]]]
```

Independent `predict/3` calls arriving within the window are coalesced into a single backend
invocation. Watch `[:ml_serve, :batch, :flush]` — a healthy setup flushes mostly on `:full`.

## Telemetry

```text
[:ml_serve, :prediction, :start | :stop | :exception]
[:ml_serve, :model, :load]  [:ml_serve, :model, :unload]
[:ml_serve, :cache, :hit]   [:ml_serve, :cache, :miss]
[:ml_serve, :batch, :flush]
```

Measurements include `duration`, `queue_duration`, `inference_duration` and `batch_size`; metadata
carries `model`, `version`, `backend`, `cached?` and `canary?`.

```elixir
MLServe.Telemetry.Logger.attach(level: :info)          # zero-dependency visibility
MLServe.Telemetry.Metrics.metrics()                    # Telemetry.Metrics / LiveDashboard
```

Because `version` and `canary?` ride along on every event, comparing a canary against the incumbent
needs no extra instrumentation — which is what makes a promote/rollback decision possible.

## Backends

MLServe ships two dependency-free backends: `MLServe.Backend.Function` (wrap any function or MFA)
and `MLServe.Backend.Static` (fixed result — swap it in `config/test.exs` to test your app without
a model).

Real ML runtimes are **guides with complete implementations**, not dependencies, so the core stays
at one runtime dep and backends version independently:

- [Nx and `Nx.Serving`](guides/creating-a-backend.md#nx)
- [Bumblebee](guides/creating-a-backend.md#bumblebee)
- [ONNX via Ortex](guides/creating-a-backend.md#onnx)
- [A Python model server over a port](guides/creating-a-backend.md#python)

## Phoenix integration

Phoenix is not a dependency. It does not need to be:

```elixir
defmodule MyAppWeb.MLController do
  use MyAppWeb, :controller

  def predict(conn, params) do
    case MLServe.predict(:fraud_detection, params) do
      {:ok, result} -> json(conn, result)
      {:error, reason} -> conn |> put_status(status_for(reason)) |> json(%{error: inspect(reason)})
    end
  end
end
```

See the [Phoenix guide](guides/phoenix-integration.md) for status-code mapping, LiveDashboard
metrics and readiness probes, and the [Oban guide](guides/oban-integration.md) for asynchronous
inference with retry semantics driven by `MLServe.Error.retryable?/1`.

## Errors

Every function returns `{:ok, result}` or `{:error, reason}`; bang variants raise `MLServe.Error`.

| Reason | Meaning |
| --- | --- |
| `:model_not_found` | Not registered under that name or version |
| `:model_not_ready` | Registered, but loading, draining or failed |
| `:timeout` | The deadline passed before inference completed |
| `:overloaded` | At the `:max_concurrency` limit |
| `{:invalid_input, reason}` | Rejected by a `:preprocess` hook |
| `{:batch_too_large, max}` | Batch exceeded `:max_batch_size` |
| `{:backend_error, %MLServe.BackendError{}}` | The backend raised — exception and stacktrace preserved |
| `{:load_failed, reason}` | The model could not be loaded |

Backend failures are never swallowed: a raise is caught only to attach the model, version,
callback and stacktrace before surfacing it.

## Security

- Model paths are validated against a configured `:model_root`, with `..` traversal and
  symlink escape rejected, plus existence, readability and size checks.
- Optional `checksum: {:sha256, "..."}` verification on load.
- Backend modules are verified to implement `MLServe.Model` at load time, not assumed.
- **MLServe never calls `binary_to_term/1`, `Code.eval_*`, or loads a NIF from a model artifact.**
  A model file is data. Supplying one is not a way to execute code.

## Documentation

Full documentation is on [HexDocs](https://hexdocs.pm/ml_serve), including guides for
[Getting Started](guides/getting-started.md), [Architecture](guides/architecture.md),
[Creating a Model Backend](guides/creating-a-backend.md),
[Running Inference](guides/running-inference.md), [Batch Inference](guides/batch-inference.md),
[Concurrency](guides/concurrency.md), [Telemetry](guides/telemetry.md),
[Model Versioning](guides/model-versioning.md), [Phoenix](guides/phoenix-integration.md),
[Oban](guides/oban-integration.md) and [Production Deployment](guides/production-deployment.md).

### Examples

[`examples/`](https://github.com/jamesnjovu/ml_serve/tree/main/examples) has runnable code for
everything above — it lives on GitHub rather than in the Hex package:

* **[Seven scripts](https://github.com/jamesnjovu/ml_serve/tree/main/examples/scripts)**, one
  command each, covering concurrency, batching, caching, versioning, telemetry and the full error
  taxonomy: `elixir examples/scripts/01_quick_start.exs`
* **[Three Livebook notebooks](https://github.com/jamesnjovu/ml_serve/tree/main/examples/notebooks)**
  for the same ground, interactively.
* **[An HTTP inference service](https://github.com/jamesnjovu/ml_serve/tree/main/examples/inference_service)**
  on Bandit and Plug — models declared in configuration, both execution strategies, and a full
  mapping from MLServe's error taxonomy onto HTTP status codes.

### For AI assistants and LLM tooling

[`usage-rules.md`](https://github.com/jamesnjovu/ml_serve/blob/main/usage-rules.md) is a condensed,
machine-readable summary of the public API and its constraints — including the mistakes that are
easy to make — and [`llms.txt`](https://github.com/jamesnjovu/ml_serve/blob/main/llms.txt) points at
the full documentation set. Point your coding agent at either; both ship inside the Hex package.

## Contributing

Issues and pull requests are welcome. Before submitting:

```bash
mix lint    # format --check-formatted + compile --warnings-as-errors + credo --strict
mix test
mix dialyzer
```

## License

Released under the [MIT License](LICENSE). Copyright © 2026 James Njovu.
