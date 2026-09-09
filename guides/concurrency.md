# Concurrency

MLServe's central design decision is that **there is no single right way to run a model
concurrently**, because ML backends genuinely differ. This guide explains the two strategies, how
to pick, and how to size a pool.

## The problem with "just use a pool"

The obvious design is a pool of GenServers, each holding the model, with requests round-robined
across them. It is the right answer for some backends and actively wrong for others.

On the BEAM, sending a term to another process **copies it**. Send a tensor into a worker's
mailbox and you copy it in, then copy the result back. For an `Nx.Serving` — which is already safe
to call concurrently and does its own internal batching — those two copies are pure waste, and the
pool is a bottleneck protecting something that needed no protection.

Meanwhile an ONNX Runtime session or a Python port genuinely cannot be shared, and calling one
concurrently produces corrupted results or a crash.

So the backend declares which it is.

## Two strategies

```elixir
@impl true
def capabilities, do: %{concurrency: :shared, load: :once}
```

### `:shared` — runs in the caller

- `predict/2` executes **in the calling process**
- State is read from `:persistent_term`, whose reads do **not** copy
- **Zero worker processes are started**, whatever `workers:` says
- No message passing, no mailbox, no queue

Use for: `Nx.Serving`, Bumblebee, pure Elixir models, HTTP calls to a remote inference service —
anything with no session to serialise access to.

```elixir
# 200 concurrent callers, 200 concurrent inferences, no MLServe process involved
1..200 |> Task.async_stream(&MLServe.predict(:embeddings, &1)) |> Enum.to_list()
```

Bound the concurrency with `:max_concurrency` if you need to — it uses an atomic counter, so it
still involves no processes:

```elixir
models: [remote_scorer: [max_concurrency: 64]]
```

> #### Timeouts do not apply {: .warning}
>
> There is no other process to walk away from, so `:timeout` is not enforced for `:shared`
> backends. Use `:exclusive` if you need MLServe to bound inference time.

### `:exclusive` — runs in a pooled worker

- State lives in `MLServe.Worker` processes
- One request per worker at a time; concurrency equals the pool size
- Requests carry a deadline the worker checks before running anything

Use for: ONNX Runtime sessions, Python ports, anything stateful or not thread-safe.

```elixir
models: [fraud_detection: [workers: 4, selection: :round_robin]]
```

## When the model loads

Crossed with the above:

| `load` | `load/1` is called | Use for |
| --- | --- | --- |
| `:once` (default) | Once; the state term is handed to every worker | NIF-resource-backed models |
| `:per_worker` | Once per worker | Ports, per-worker sessions |

`:once` is the default because most real ML state is a **NIF resource**: the Elixir term is a
cheap handle, and the GPU or heap memory behind it is shared by every process holding a copy of
the handle. Loading a 2 GB model separately into eight workers is an out-of-memory crash, not a
pool.

Choose `:per_worker` only when each worker genuinely needs its own OS-level resource:

```elixir
models: [
  recommender: [
    backend: MyApp.Backends.PythonPort,
    workers: 4,               # 4 Python processes, 4 independent ports
    restart_on_error: true
  ]
]
```

## Sizing the pool

The default is `System.schedulers_online()`, which is a reasonable starting point for CPU-bound
models and usually wrong for everything else.

| Backend | Guidance |
| --- | --- |
| CPU inference, single-threaded | `workers: System.schedulers_online()` |
| CPU inference, internally threaded (BLAS, ORT with intra-op threads) | `workers: 1..2`. The engine is already using every core; more workers just fight over them. |
| GPU inference | `workers: 1..2` plus `batching:`. A GPU is one device; queuing at it is the batcher's job, not the pool's. |
| Python port | One per available CPU core, minus headroom |
| Remote HTTP service | `:shared` with `max_concurrency:`, not a pool |

The most common mistake is a large pool in front of an engine that is already parallel internally.
Eight workers each asking a 14-thread BLAS to use every core produces contention, not throughput.

## Selection strategies

```elixir
models: [fraud_detection: [workers: 4, selection: :least_loaded]]
```

| Strategy | |
| --- | --- |
| `:round_robin` (default) | An atomic fetch-and-add. Exactly fair, constant cost, predictable. |
| `:least_loaded` | Samples two workers and takes the shorter mailbox — "power of two choices". Most of the benefit of a full scan at constant cost, and it avoids the herd behaviour of everyone piling onto the single least-loaded worker. |
| `:random` | Uniform. Useful when request costs vary wildly and you want no pattern at all. |

`:round_robin` is right when requests cost roughly the same. `:least_loaded` earns its keep when
they do not — one slow request should not make three later ones wait behind it.

A worker restarting leaves a momentary gap in the registry; dispatch tries the next couple of
indices rather than failing, so a restart is invisible to callers.

## Isolation

Every loaded model version is its own supervision subtree, so:

```elixir
# Model A's backend is crash-looping. Model B does not care.
MLServe.predict(:model_b, input)   #=> {:ok, result}
```

A slow model does not block a fast one either — they share no process, no queue and no lock. The
only thing they share is the ETS catalog, which is read-only on the hot path.

## Admission control

```elixir
models: [fraud_detection: [workers: 4, max_concurrency: 64]]
```

Past the limit, requests are rejected immediately with `{:error, :overloaded}` rather than queued.
Rejecting fast is almost always better than queueing slowly: a caller that gets `:overloaded` in
microseconds can fall back, shed, or return a cached answer, while a caller stuck in a queue holds
a connection and a process for the full timeout and then fails anyway.

The counter is atomic and process-free. Under a race two callers can both observe `max - 1` and
both proceed, so the limit is approximate at the boundary — the right trade, since an exact
limiter would need serialisation through a process, which is the very bottleneck the limit exists
to prevent.

## Deadlines and load shedding

Requests carry an absolute deadline, checked by the worker **before** the backend runs. Expired
work is dropped, not executed.

Under overload this is the difference between recovery and collapse. Without it, a worker works
through a queue of requests whose callers have all timed out, each new arrival waiting longer than
the last, producing results nobody will read — while new work piles up behind it.

## Measuring it

```elixir
MLServe.model_status(:fraud_detection)
#=> {:ok, %{in_flight: 3, requests: 154_223, errors: 12, workers: 4, ...}}
```

| Signal | Meaning |
| --- | --- |
| `in_flight` close to `workers` | Saturated. Add workers, or batch. |
| `in_flight` close to `max_concurrency` | Shedding. Check `:overloaded` error rates. |
| `queue_duration` climbing | Requests waiting for a worker — the pool is the constraint. |
| `inference_duration` climbing | The model itself got slower, or the machine is contended. |

`queue_duration` and `inference_duration` are separate measurements on
`[:ml_serve, :prediction, :stop]` precisely so you can tell those last two apart. See
[Telemetry](telemetry.md).
