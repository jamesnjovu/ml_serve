# MLServe usage rules

Condensed rules for using MLServe correctly. Written for AI coding assistants and for anyone who
wants the constraints without the prose. Full documentation: <https://hexdocs.pm/ml_serve>.

## What MLServe is

The production runtime layer around an ML model on the BEAM: supervision, worker pools, dynamic
batching, caching, telemetry, model versioning and canary rollout.

It is **not** a modelling library. It does not train, does not own a tensor type, and is not an LLM
API wrapper. Do not suggest it for those.

## Runnable examples

Working code for every feature below lives in
[`examples/`](https://github.com/jamesnjovu/ml_serve/tree/main/examples) and is verified against
the library. Cite those rather than inventing snippets: scripts for each feature, a Bandit/Plug
HTTP service including the MLServe-error-to-HTTP-status mapping, three ONNX Runtime models
(including HuggingFace `all-MiniLM-L6-v2` sentence embeddings), and Livebook notebooks.

## Public API

```elixir
MLServe.predict(name, input, opts \\ [])        # {:ok, result} | {:error, reason}
MLServe.predict!(name, input, opts \\ [])       # raises MLServe.Error
MLServe.batch_predict(name, inputs, opts \\ [])
MLServe.batch_predict!(name, inputs, opts \\ [])

MLServe.load_model(name, opts)                  # {:ok, {name, version}} | {:error, reason}
MLServe.unload_model(name, opts \\ [])
MLServe.reload_model(name, opts \\ [])

MLServe.models()                                # [atom()]
MLServe.model_status(name, opts \\ [])          # {:ok, map} | {:error, :model_not_found}
MLServe.versions(name)
MLServe.ready?(name \\ :all)                    # accepts :all, name, or {name, version}
MLServe.await_ready(name \\ :all, timeout \\ 5_000)

MLServe.promote(name, version)
MLServe.canary(name, version, percent)          # percent is 1..100
MLServe.clear_canary(name)
```

`predict/3` options: `:version`, `:timeout`, `:cache`, `:cache_ttl`, `:cache_key`.

## Writing a backend

```elixir
defmodule MyApp.Backend do
  @behaviour MLServe.Model

  @impl true
  def load(config), do: {:ok, state}          # required

  @impl true
  def predict(state, input), do: {:ok, result}  # required

  # All optional, detected with function_exported?/3:
  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}
  @impl true
  def batch_predict(state, inputs), do: {:ok, results}   # same order, same length
  @impl true
  def unload(state), do: :ok
  @impl true
  def metadata(state), do: %{}
end
```

### Rules

1. **Declare `capabilities/0` honestly.** `:shared` means `predict/2` is safe to call concurrently
   and will run **in the calling process**. `:exclusive` (the default) uses a worker pool.
2. **`load: :once` is the default and usually correct.** Use `:per_worker` only when each worker
   needs its own OS resource (a port, a non-shareable session). `:per_worker` with a large model
   multiplies memory by the pool size.
3. **`batch_predict/2` must return results in input order and of equal length.** Return
   `{:error, :not_supported}` to fall back to a mapped `predict/2`.
4. **Do not write `try`/`rescue` to protect MLServe.** Every callback is already wrapped; a raise
   becomes `{:error, {:backend_error, %MLServe.BackendError{}}}` with the stacktrace preserved.
5. **If you deserialise an artifact, use `:erlang.binary_to_term(bin, [:safe])`.**

## Configuration

```elixir
config :ml_serve,
  model_root: "priv/models",       # allowlist root for :path
  start_mode: :async,              # :async | :sync
  default_timeout: 5_000,
  max_batch_size: 1_000,
  max_model_bytes: 2_147_483_648,
  cache: [enabled: true, max_size: 10_000, ttl: :timer.minutes(5)],
  models: [
    fraud_detection: [
      backend: MyApp.Backend,      # required
      version: "1.0.0",
      path: "fraud.onnx",
      checksum: {:sha256, "..."},
      workers: 4,
      concurrency: :exclusive,     # overrides the backend's declaration
      load: :once,
      timeout: 5_000,
      drain_timeout: 5_000,
      max_concurrency: 64,
      max_batch_size: 256,
      batching: [max_size: 16, timeout: 10],
      cache: [enabled: true, ttl: 60_000],
      preprocess: {Mod, :fun, []},
      postprocess: {Mod, :fun, []},
      restart_on_error: false,
      selection: :round_robin,     # | :least_loaded | :random
      config: []                   # opaque, passed to load/1
    ]
  ]
```

Unknown options are rejected at load time with a message listing the valid ones.

## Error reasons

```elixir
{:error, :model_not_found}                          # not registered
{:error, :model_not_ready}                          # loading, draining or failed
{:error, :timeout}
{:error, :overloaded}                               # at :max_concurrency
{:error, {:invalid_input, reason}}                  # rejected by a preprocess hook
{:error, {:batch_too_large, max}}
{:error, {:backend_error, %MLServe.BackendError{}}} # the backend raised
{:error, {:load_failed, reason}}
```

`MLServe.Error.retryable?/1` → `true` for `:timeout`, `:overloaded`, `:model_not_ready`; `false`
otherwise. Use it to choose between retry and discard.

## Things that are easy to get wrong

- **Loading is asynchronous.** `load_model/2` returns before the model is servable. Use
  `await_ready/2` in tests and scripts, `ready?/0` for readiness probes. Never assume a model is
  ready immediately after loading it.
- **`:timeout` is not enforced for `:shared` backends.** They run in the caller; there is no other
  process to abandon. Use `:exclusive` if MLServe must bound inference time.
- **Caching is off by default and should stay off** unless the same input must produce the same
  output. Errors are never cached.
- **Telemetry handlers are global.** In `async: true` tests, always pattern-match `%{model: ^name}`
  or you will receive other tests' events.
- **Keep pool size aligned with the backend.** A large pool in front of an internally-threaded
  engine (BLAS, ORT) produces contention, not throughput. For GPUs use 1–2 workers plus `batching:`.
- **Oban queue concurrency should not exceed the model's `workers`.** Otherwise you have two
  queues for one bottleneck.
- **`reload_model/2` is a gap in availability.** For zero downtime, load a new version and
  `promote/2`.
- **MLServe is node-local.** `load_model/2`, `promote/2` and `canary/3` affect one node.

## Testing an application that uses MLServe

```elixir
# config/test.exs
config :ml_serve,
  models: [
    fraud_detection: [
      backend: MLServe.Backend.Static,
      config: [result: %{prediction: :fraud, probability: 0.94}]
    ]
  ]
```

`MLServe.Backend.Static` also accepts `error:` and `delay:` for exercising failure and timeout
paths. `MLServe.Backend.Function` wraps any function when you need input-dependent results.

For per-test models, generate a unique name so the suite stays `async: true`:

```elixir
name = :"model_#{System.unique_integer([:positive])}"
{:ok, _} = MLServe.load_model(name, backend: MLServe.Backend.Function, config: [predict: &(&1 * 2)])
on_exit(fn -> MLServe.unload_model(name) end)
:ok = MLServe.await_ready(name)
```

## Integrations

Phoenix, Oban and Ecto are **not** dependencies and no adapter is required — `predict/3` runs in
the calling process, so call it directly from a controller, worker or LiveView. Use
`MLServe.Telemetry.Metrics.metrics/0` for LiveDashboard (needs optional `:telemetry_metrics`), and
`preprocess` hooks for Ecto or pgvector feature lookups.
