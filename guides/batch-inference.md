# Batch Inference

Two different things share the word "batch", and MLServe supports both because they solve
different problems.

|  | Shape | API |
| --- | --- | --- |
| **Explicit batching** | One caller, many inputs | `MLServe.batch_predict/3` |
| **Dynamic batching** | Many concurrent callers, one input each | `batching:` configuration |

## Explicit batching

```elixir
MLServe.batch_predict(:fraud_detection, [features_a, features_b, features_c])
#=> {:ok, [result_a, result_b, result_c]}
```

Results come back in input order. When the backend implements
`c:MLServe.Model.batch_predict/2` this is a **single** backend call — one round-trip and one
vectorised computation instead of N, which is the entire point.

Without that callback, MLServe maps `predict/2` over the inputs and stops at the first error,
since a batch result is all-or-nothing.

Check which you are getting:

```elixir
MLServe.model_status(:fraud_detection)
#=> {:ok, %{native_batching: true, ...}}
```

### Implementing it

```elixir
@impl true
def batch_predict(state, inputs) do
  tensor = inputs |> Enum.map(&encode/1) |> Nx.tensor(type: :f32)
  {probabilities} = Ortex.run(state.session, {tensor})
  {:ok, probabilities |> Nx.to_flat_list() |> Enum.map(&decode/1)}
end
```

Two rules, both enforced: results must be **in input order** and of **equal length**. A backend
returning the wrong count gets a clear `MLServe.BackendError` naming the mismatch rather than
handing callers each other's results.

A backend that exports `batch_predict/2` but cannot batch a particular model returns
`{:error, :not_supported}` to fall back to the mapped path.

### Limits

```elixir
config :ml_serve, max_batch_size: 1_000
models: [fraud_detection: [max_batch_size: 256]]
```

Oversized batches are rejected before any work happens:

```elixir
MLServe.batch_predict(:fraud_detection, Enum.to_list(1..5_000))
#=> {:error, {:batch_too_large, 256}}
```

This is a real safety limit, not a formality: an unbounded batch is an unbounded allocation, and
one request can otherwise exhaust the memory of the node.

## Dynamic batching

The far more common production shape is many independent callers, each with a single input,
arriving within milliseconds of each other. A GPU that processes 32 rows in barely more time than
one row is being wasted by a pool feeding it one row at a time.

```elixir
models: [
  fraud_detection: [
    backend: MyApp.Backends.ONNX,
    workers: 2,
    batching: [max_size: 16, timeout: 10]
  ]
]
```

Nothing changes at the call site. Independent `predict/3` calls are transparently coalesced:

```elixir
# 16 separate Phoenix requests, arriving within 10ms → one backend call
MLServe.predict(:fraud_detection, features)
```

### The window

A batch flushes when either trigger fires:

- `:max_size` inputs have accumulated → `reason: :full`
- `:timeout` milliseconds have passed **since the first input** → `reason: :timeout`

Timing from the first input rather than the last bounds the added latency at `:timeout` for every
caller. A sliding window timed from the most recent arrival can starve the earliest caller
indefinitely under steady traffic.

### Tuning it

Watch `[:ml_serve, :batch, :flush]`:

```elixir
:telemetry.attach("batch-watch", [:ml_serve, :batch, :flush], fn _e, m, meta, _c ->
  IO.puts("#{meta.model}: #{m.size} rows on #{meta.reason}")
end, nil)
```

| What you see | What it means |
| --- | --- |
| Mostly `:full` | Healthy. Batches fill before the window closes. |
| Mostly `:timeout`, small sizes | The window is longer than your arrival rate justifies — you are adding latency for batches that never fill. Reduce `:timeout`, or drop batching entirely. |
| Mostly `:full`, high `wait_duration` | Workers are saturated. Add workers or a bigger `:max_size`. |

Start with `max_size` matching the batch size your model was compiled for, and `timeout` at the
latency budget you can spare — 5–20ms is typical for an interactive request path.

### Why it does not block

Running inference inside the batcher would stop it accumulating the *next* batch for the whole
duration of the current one, serialising exactly what it was built to parallelise. Instead a flush
hands the batch to a task under `MLServe.TaskSupervisor`, which calls a worker and replies to
every caller with `GenServer.reply/2`. The batcher stays responsive throughout.

### Backpressure

In-flight batches are capped at the model's worker count. Beyond that, flushes wait — there is no
free worker to take them, and queueing more would only build a backlog of work whose callers will
have timed out by the time it runs. That cap **is** the backpressure.

Expired requests are dropped from a batch at flush time rather than being sent to the backend.

### Batching and `batch_predict/3` together

`MLServe.batch_predict/3` bypasses the batcher and calls the backend directly. That is correct:
you already have a full batch, so there is nothing to wait for.

## Choosing

| Situation | Use |
| --- | --- |
| Nightly scoring of a table | `batch_predict/3` |
| An Oban job scoring 10k rows | `batch_predict/3`, chunked to `max_batch_size` |
| A Phoenix endpoint under concurrent load | `batching:` config |
| A GPU-backed model | `batching:` config, almost always |
| A cheap CPU model, low traffic | Neither — the coordination costs more than it saves |

Batching is not free. It adds up to `:timeout` milliseconds of latency to every request. Enable it
when the per-call overhead of your backend genuinely dominates the per-row cost, which is the case
for GPUs and rarely the case for a small CPU model.

## Chunking large jobs

```elixir
def score_all(rows) do
  rows
  |> Stream.chunk_every(256)
  |> Task.async_stream(
    fn chunk -> MLServe.batch_predict(:fraud_detection, chunk, timeout: 30_000) end,
    max_concurrency: 4,
    timeout: 35_000
  )
  |> Enum.reduce({:ok, []}, fn
    {:ok, {:ok, results}}, {:ok, acc} -> {:ok, acc ++ results}
    {:ok, {:error, reason}}, _acc -> {:error, reason}
    {:exit, reason}, _acc -> {:error, {:exit, reason}}
  end)
end
```

Keep `max_concurrency` at or below the model's worker count — going higher just queues work that
`:max_concurrency` may then reject as `:overloaded`.
