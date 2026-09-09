# Architecture

MLServe is a runtime layer, not a modelling library. This guide explains what it puts around your
model and why each piece is shaped the way it is.

## The supervision tree

```text
MLServe.Supervisor                     (:rest_for_one)
├── MLServe.Registry                   Registry, :unique, partitions: schedulers
├── MLServe.ModelRegistry              GenServer — owns the catalog ETS table
├── MLServe.Cache                      GenServer — owns the cache ETS table + TTL sweeper
├── MLServe.TaskSupervisor             Task.Supervisor — batch fan-out
├── MLServe.ModelSupervisor            DynamicSupervisor
│     └── MLServe.ModelInstance        (:rest_for_one) one per loaded {name, version}
│           ├── MLServe.ModelServer    lifecycle, status, graceful drain
│           ├── MLServe.WorkerSupervisor  (:one_for_one) omitted for shared backends
│           │     └── MLServe.Worker × N
│           └── MLServe.Batcher        only when dynamic batching is configured
└── MLServe.Bootstrap                  transient Task — loads configured models, then exits
```

### Why `:rest_for_one` at the top

`MLServe.ModelRegistry` owns the catalog ETS table, and **ETS tables die with their owner**. If
that process restarts, every route in the system has just evaporated. Any model subtree still
running would be serving traffic the registry no longer knows about, and the next `predict/3`
would report `:model_not_found` for a model whose workers are demonstrably alive.

`:rest_for_one` restarts everything *after* the registry, which is the only consistent answer.
`:one_for_one` would leave orphans.

The relationship is not symmetric: a crashed model subtree tells us nothing about the registry, so
nothing above it restarts. That asymmetry is exactly what per-model isolation means.

### Why a subtree per model version

The obvious design puts one worker supervisor at the top and pools everything. Per-model subtrees
are better on three counts:

1. **Restart intensity is scoped.** A backend that crash-loops burns through *its own* budget and
   takes down *its own* model. A shared supervisor would let one bad model kill healthy ones.
2. **Unload is a single `terminate_child`.** Workers, batcher and server go together.
3. **Two versions coexist as siblings**, which is what makes canary rollout and drain-on-unload
   possible at all.

Inside the subtree, `:rest_for_one` and the child order are load-bearing: `MLServe.ModelServer`
owns the loaded backend state the workers were handed. If it dies, their state is stale, so they
must restart behind it. A crashed worker has no such implication for the server.

## The prediction path

```text
caller process
  route lookup                     1 lock-free :ets.lookup
  status + canary roll             pure, in-process
  admission control                atomic :counters
  ┌ telemetry span opens
  │ cache lookup                   opt-in
  │ preprocess hook                caller's own scheduler time
  │ dispatch ──────────────┬─ :shared    → :persistent_term.get + backend, in this process
  │                        ├─ batching   → GenServer.call(Batcher)
  │                        └─ :exclusive → :ets.update_counter + Registry.lookup + call(Worker)
  │ postprocess hook
  │ cache write                    successes only
  └ telemetry span closes
```

**There is no MLServe process on this path** other than the worker that runs inference, and for
shared backends not even that.

### Why the registry is never called

`predict/3` runs in whatever process handles the request. If finding a model meant a
`GenServer.call` to the registry, that one process would serialise the entire node's inference
traffic *before any inference happened* — the classic accidental bottleneck.

So the registry is a GenServer that **owns** a `:protected` ETS table with `read_concurrency: true`
but is never in the read path:

- reads (`route/2`, `model_status/2`) are direct `:ets.lookup/2` in the caller
- writes (register, promote, canary, unregister) serialise through the GenServer

Serialising writes is free: they happen at deploy time, not per request.

### Route versus spec

Two rows are stored per model. `MLServe.ModelSpec` is the full configuration, including the
backend's `:config` — which may hold closures, lookup tables or large keyword lists. Reading it out
of ETS copies the whole thing into the calling process.

So the hot path reads a `MLServe.Route` instead: a deliberately slim struct of atoms, small
integers and references. Nothing that grows with model size ever enters it.

### Why hooks run in the caller

`preprocess` and `postprocess` run *before dispatch*, in the calling process. A feature-store
lookup against Postgres is I/O; occupying a scarce GPU worker slot while it happens would be
throwing away the exact resource the pool exists to protect.

## Two concurrency models

This is the central design decision, and where a naive "pool of GenServers" loses badly on the
BEAM: sending a large tensor into a worker's mailbox **copies it**, then copies the result back.
For an `Nx.Serving` — already concurrency-safe, already batching internally — that copy is pure
waste and the pool is a bottleneck protecting nothing.

So a backend declares how it may be called:

| `concurrency` | Execution | State storage |
| --- | --- | --- |
| `:shared` | In the calling process. **Zero worker processes.** | `:persistent_term` — reads do not copy |
| `:exclusive` | In a pooled `MLServe.Worker` | Worker state, or a shared handle |

Storage is split deliberately:

- **Routing table → ETS** (`read_concurrency`). Small terms, read constantly, written rarely.
- **Shared backend state → `:persistent_term`**. Reads are copy-free, which matters when the term
  is a model handle; writes trigger a global scan, which is fine because they happen only at load
  and unload.

Crossed with when the model loads:

| `load` | Meaning |
| --- | --- |
| `:once` (default) | `load/1` runs once, the state term is handed to every worker |
| `:per_worker` | `load/1` runs per worker, each with independent state |

`:once` is the default because most real ML state is a NIF resource: the term is a cheap handle
and the GPU or heap memory behind it is shared. Loading a 2 GB model separately into eight workers
is an out-of-memory crash, not a pool.

## Deadlines and load shedding

Every request carries an absolute monotonic deadline, and a worker checks it **before** invoking
the backend, dropping already-expired work.

This matters under overload. A `GenServer.call` timeout only abandons the *caller's* side; the
worker still grinds through the entire queue, each item arriving later than the last, serving
results nobody is waiting for. Checking the deadline at the front of the queue converts a death
spiral into load shedding.

`:max_concurrency` adds admission control on top, using an atomic counter — no process, no lock.
Past the limit, requests are rejected immediately with `{:error, :overloaded}` rather than queued.

> #### Shared backends and timeouts {: .warning}
>
> A `:shared` backend runs in your process, so there is no other process to walk away from and
> `:timeout` cannot be enforced. That is the same contract as calling any function directly. If you
> need MLServe to bound inference time, use `concurrency: :exclusive`.

## Failure handling

Every backend invocation is wrapped. A raise, throw or exit becomes
`{:error, {:backend_error, %MLServe.BackendError{}}}` carrying the original exception **and
stacktrace**, and is simultaneously reported on `[:ml_serve, :prediction, :exception]`.

Catching is only ever used to attach context — the model, version, callback and stacktrace — never
to hide the failure. A model raising `ArgumentError` deep inside a tensor operation is a bug you
need the line number for; `{:error, :something_went_wrong}` would throw that away.

By default the worker survives, because most backend errors are about the input, not the state.
`restart_on_error: true` opts into crashing the worker instead, for backends whose state may be
corrupt after a failure — a port that died, a session left mid-transaction.

## Dependencies

One runtime dependency: `:telemetry`. Configuration validation is hand-rolled rather than pulling
in a schema library, and Phoenix, Oban, Ecto and Plug are documented integrations rather than
dependencies. Real ML runtimes ship as guides containing complete implementations, so the core
stays lightweight and backends version independently.
