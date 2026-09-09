# Changelog

All notable changes to the MLServe library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-09-09

### Fixed

- **`MLServe.Security.digest/2` raised on Elixir 1.14 through 1.16**, which broke every
  `:checksum` verification on three of the five Elixir versions this library supports. The
  chunked read went through `File.stream!/2`, whose second argument only came to mean a byte
  count in Elixir 1.17 — before that it is still `modes`, so an integer raised
  `FunctionClauseError` in `File.normalize_modes/2`. There is no single `File.stream!` call that
  is correct on 1.14 and free of a contract violation on 1.18, so hashing now reads the file with
  `File.open/2` and `:file.read/2`, which have been stable across every supported version.
  Streaming behaviour and peak memory are unchanged.

- **`MLServe.Security.validate_path/2` reported `:enoent` instead of `:outside_root`** for a
  traversal whose target happened not to exist, making the result depend on the filesystem rather
  than on the path. Containment is now decided on the expanded path before the file is touched
  and re-checked after symlink resolution, so a path outside `:model_root` is always rejected as
  `:outside_root` — and MLServe no longer stats a caller-supplied path outside the root.

## [0.1.0] - 2026-08-24

Initial release.

### Added

- **`MLServe.Model` behaviour.** Two required callbacks — `load/1` and `predict/2` — with
  `batch_predict/2`, `unload/1`, `metadata/1` and `capabilities/0` optional and detected via
  `function_exported?/3`. The smallest useful backend is two functions.

- **Two execution strategies, declared by the backend.** `concurrency: :shared` runs `predict/2`
  **in the calling process** with state read from `:persistent_term`, starting no worker processes
  and copying no tensors between mailboxes — the right shape for `Nx.Serving`, Bumblebee, pure
  functions and remote services. `concurrency: :exclusive` uses a supervised worker pool, for ONNX
  sessions, ports and anything not thread-safe. Crossed with `load: :once | :per_worker`, so a
  NIF-resource model is loaded once and shared rather than once per worker.

- **Per-model supervision subtrees.** Each loaded `{name, version}` is its own `:rest_for_one`
  subtree under a `DynamicSupervisor`, so a backend that crash-loops exhausts its own restart
  budget and marks itself `:failed` while every other model keeps serving.

- **Lock-free dispatch.** `MLServe.ModelRegistry` owns a `:protected`, `read_concurrency` ETS
  catalog but is never in the read path: `predict/3` resolves a model with a direct `:ets.lookup/2`
  in the caller. The hot path contains no MLServe process other than the worker running inference.

- **Asynchronous model loading with backoff retry.** `load_model/2` returns once the model is
  registered; loading proceeds under `handle_continue/2` so a thirty-second load never blocks
  application boot. Failures retry with exponential backoff before the model is marked `:failed`,
  with the reason preserved in `model_status/2`. `ready?/1` and `await_ready/2` support readiness
  probes and tests.

- **Dynamic batching.** Optional per-model `batching: [max_size: 16, timeout: 10]` coalesces
  independent concurrent `predict/3` calls into one backend invocation. The batcher never blocks on
  inference — flushes are handed to a supervised task — and in-flight batches are capped at the
  worker count, which is the backpressure. `[:ml_serve, :batch, :flush]` reports `:full` versus
  `:timeout` so the window can be tuned.

- **Explicit batching.** `batch_predict/3` calls the backend's `batch_predict/2` when exported, and
  falls back to a mapped `predict/2` otherwise. Result count and order are validated.

- **Deadlines and load shedding.** Requests carry an absolute monotonic deadline that workers check
  *before* invoking the backend, so work whose caller has already timed out is dropped rather than
  run. `max_concurrency` adds process-free admission control via atomic counters, returning
  `{:error, :overloaded}` rather than queueing.

- **Model versioning, canary rollout and graceful drain.** Models are keyed `{name, version}` and
  run side by side. `canary/3` routes a percentage of unpinned traffic to a candidate, rolled
  per-request in the caller with no coordination point; `promote/2` flips the default with a single
  ETS write; `unload_model/2` drains in-flight requests before terminating, reporting stragglers as
  the `drained` measurement. Telemetry carries `version` and `canary?`, which is what makes the
  promote-or-roll-back decision measurable.

- **`preprocess` / `postprocess` hooks.** Per-model `{module, function, args}` hooks that run in the
  **calling process, before dispatch** — so a feature-store or pgvector query never occupies a GPU
  worker slot. One mechanism covers feature enrichment, tensor conversion, label decoding, and
  input validation (a hook returning `{:error, {:invalid_input, reason}}` rejects the request).

- **Optional inference cache.** Off by default, because caching is only correct when the same input
  must produce the same output. ETS-backed with TTL, lazy and swept expiry, and a `max_size` bound.
  Keys use `:erlang.term_to_binary(input, [:deterministic])` so equal maps hash equally; a custom
  `:cache_key` avoids hashing large inputs. Errors are never cached, and entries are invalidated
  when a model version is unloaded.

- **Telemetry.** `[:ml_serve, :prediction, :start | :stop | :exception]` as a standard span, plus
  `[:ml_serve, :model, :load]`, `[:ml_serve, :model, :unload]`, `[:ml_serve, :cache, :hit | :miss]`
  and `[:ml_serve, :batch, :flush]`. `queue_duration` and `inference_duration` are reported
  separately so a rise in latency distinguishes "the pool is too small" from "the model got
  slower". A backend that *returns* an error produces a `:stop` with `result: :error`, not an
  `:exception` — an expected rejection is not a crash.

- **`MLServe.Telemetry.Logger`** for dependency-free visibility, and **`MLServe.Telemetry.Metrics`**
  returning `Telemetry.Metrics` definitions for LiveDashboard when the optional `:telemetry_metrics`
  dependency is present, and `[]` when it is not.

- **Structured errors.** `MLServe.Error` with a `:type`, and `MLServe.BackendError` carrying the
  backend, callback, kind, reason **and stacktrace** when a backend raises, throws or exits.
  Failures are never swallowed — catching is used only to attach context before surfacing.
  `MLServe.Error.retryable?/1` separates transient conditions from permanent ones, which is what
  the Oban guide uses to choose between `:snooze` and `:discard`.

- **Artifact security.** `MLServe.Security` validates model paths against a configured
  `:model_root`, rejecting `..` traversal and symlink escape (symlinks are resolved *before* the
  containment check), and enforcing existence, readability and a size limit. Optional
  `checksum: {:sha256, hex}` verification streams the file rather than reading it into memory, and
  compares digests in constant time. Backend modules are verified to implement `MLServe.Model` at
  load time. **The core never calls `binary_to_term/1`, `Code.eval_*`, or loads a NIF from a model
  artifact** — a model file is data.

- **Built-in backends.** `MLServe.Backend.Function` wraps any function or MFA (useful for rules
  engines and glue, and the workhorse of the test suite); `MLServe.Backend.Static` returns a fixed
  result, so applications can test against MLServe's real routing without a model file or an ML
  runtime.

- **Eleven guides**, including complete backend implementations for Nx, `Nx.Serving`, Bumblebee,
  ONNX via Ortex, a Python model server over a port, and a remote HTTP service — shipped as
  documentation rather than dependencies so the core stays at one runtime dependency and backends
  version independently.

### Robustness invariants

Found and fixed by the suite's own chaos tests, and asserted so they cannot regress:

- **A worker that dies mid-request never kills its caller.** Every exit from a worker call is
  caught and retried on another worker, so a crash during inference returns an error tuple instead
  of propagating into the Phoenix request process that made the call.
- **Worker restarts are invisible to callers.** A killed worker stays registered until its monitor
  fires, so selection checks liveness and dispatch retries a fresh worker rather than failing on a
  pid that is already gone.
- **A dead batch task frees its slot and answers its callers.** Flush tasks are monitored; without
  that, one killed task would leak an in-flight slot until the batcher wedged permanently, and its
  callers would block until their own timeouts.
- **`ready?/1` never reports ready before `model_status/2` agrees.** The status entry is written
  before the route, so a readiness probe and a status page can never disagree.
- **`[:ml_serve, :model, :load]` is emitted before a model is marked ready**, so anything observing
  readiness can rely on the load event having already fired.

### Notes

- The only runtime dependency is `:telemetry`. Phoenix, Oban, Ecto and Plug are documented
  integrations, not dependencies. Configuration validation is hand-rolled rather than pulling in a
  schema library.
- `:timeout` is **not enforced** for `concurrency: :shared` backends: they run in the calling
  process, so there is no other process to abandon. This is the same contract as calling any
  function directly, and is documented on `MLServe.predict/3`.
- MLServe is node-local by design. Each node has its own catalog, counters and cache; there is no
  distributed coordination and none is required for stateless inference.

[0.1.0]: https://github.com/jamesnjovu/ml_serve/releases/tag/v0.1.0
