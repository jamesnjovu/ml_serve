# Running Inference

Everything about calling a model: routing, options, hooks, caching and error handling.

## The call

```elixir
MLServe.predict(:fraud_detection, %{amount: 1500.50})
#=> {:ok, %{prediction: :fraud, probability: 0.94}}
```

`predict/3` runs in **your** process. There is no hop to a coordinator, no queue you join by
default — just an ETS lookup and then either a direct backend call or one `GenServer.call` to a
worker. Call it from a Phoenix controller, a LiveView, an Oban job or a `Task` without ceremony.

The bang variant raises `MLServe.Error` instead:

```elixir
MLServe.predict!(:fraud_detection, features)
#=> %{prediction: :fraud, probability: 0.94}
```

## Options

```elixir
MLServe.predict(:fraud_detection, features,
  version: "2.1.0",
  timeout: 250,
  cache: true,
  cache_ttl: :timer.minutes(5),
  cache_key: features.account_id
)
```

| Option | Default | |
| --- | --- | --- |
| `:version` | default version | Pin an exact version, bypassing canary routing |
| `:timeout` | the model's `:timeout` | Deadline in milliseconds |
| `:cache` | off | `true`, `false`, or a TTL in milliseconds |
| `:cache_ttl` | model or global TTL | TTL override |
| `:cache_key` | the input | Explicit key — cheaper than hashing a large input |

## Version routing

Without `:version`, a request routes to the model's default version — or, if a canary is active,
rolls for it. With `:version`, it goes exactly there and ignores the canary entirely.

```elixir
MLServe.predict(:fraud_detection, features)                     # default (or canary)
MLServe.predict(:fraud_detection, features, version: "2.1.0")   # pinned
```

See [Model Versioning](model-versioning.md).

## Timeouts and deadlines

The `:timeout` is an **absolute deadline that travels with the request**, not just a limit on how
long the caller waits. A worker checks it before invoking the backend and drops already-expired
work.

That distinction matters under overload. A plain `GenServer.call` timeout abandons only the
caller's side; the worker keeps grinding through a queue of requests whose callers have all given
up, each new arrival later than the last, and the system never recovers. Dropping expired work at
the front of the queue turns that death spiral into load shedding.

> #### Shared backends {: .warning}
>
> A `concurrency: :shared` backend runs in your own process. There is no other process to abandon,
> so `:timeout` is **not enforced** — the same contract as calling any function directly. Use
> `concurrency: :exclusive` if you need MLServe to bound inference time, or bound it in the
> backend.

## Preprocess and postprocess hooks

Per-model hooks that run **in the calling process, before dispatch**:

```elixir
models: [
  fraud_detection: [
    backend: MyApp.Backends.ONNX,
    preprocess: {MyApp.Features, :enrich, []},
    postprocess: {MyApp.Fraud, :decode_label, []}
  ]
]
```

A hook is `{module, function, extra_args}` or a 1-arity function, and may return `{:ok, value}`,
`{:error, reason}`, or a bare value (treated as `{:ok, value}`).

This one mechanism covers what would otherwise be several features:

### Feature store and pgvector lookups

```elixir
defmodule MyApp.Features do
  def enrich(%{account_id: id} = input) do
    case MyApp.Repo.get(MyApp.AccountFeatures, id) do
      nil -> {:error, {:invalid_input, "unknown account #{id}"}}
      features -> {:ok, Map.merge(input, Map.take(features, [:avg_amount_30d, :chargebacks]))}
    end
  end
end
```

Because hooks run in the caller, this Ecto query never occupies a worker slot. That is the point:
a GPU worker doing database I/O is the most expensive way to wait.

### Input validation

A `preprocess` hook returning `{:error, {:invalid_input, reason}}` rejects the request before it
reaches the backend:

```elixir
preprocess: fn
  %{amount: amount} = input when is_number(amount) and amount >= 0 -> {:ok, input}
  _ -> {:error, {:invalid_input, "amount must be a non-negative number"}}
end
```

```elixir
MLServe.predict(:fraud_detection, %{amount: "lots"})
#=> {:error, {:invalid_input, "amount must be a non-negative number"}}
```

### Label decoding

```elixir
postprocess: fn %{scores: scores} ->
  {:ok, %{label: Enum.max_by(scores, &elem(&1, 1)) |> elem(0)}}
end
```

Hooks apply to every element of a batch too, and one invalid element rejects the whole batch.

## Caching

Off by default, and deliberately so: caching is only correct when the same input must produce the
same output, which is false the moment a model reads a clock, a random seed, or mutable feature
state. Silently caching would turn that into a subtle correctness bug.

```elixir
MLServe.predict(:embeddings, "some text", cache: true)
```

Per model:

```elixir
models: [embeddings: [cache: [enabled: true, ttl: :timer.minutes(30)]]]
```

The default key is a SHA-256 of `:erlang.term_to_binary(input, [:deterministic])`. The
`:deterministic` flag matters — without it, large maps serialise in internal-hash order and two
equal inputs can produce different binaries, silently halving the hit rate.

For large tensor inputs, hashing the whole term costs more than the inference you are skipping.
Pass a cheap key:

```elixir
MLServe.predict(:embeddings, big_tensor, cache: true, cache_key: document_id)
```

**Errors are never cached.** A transient backend failure must not be pinned for the TTL.

Cached entries are dropped when the model version is unloaded, so a reload can never serve results
from the previous model.

## Errors

```elixir
case MLServe.predict(:fraud_detection, features) do
  {:ok, result} ->
    result

  {:error, :model_not_ready} ->
    # still loading, draining, or failed
    :unavailable

  {:error, :overloaded} ->
    # at :max_concurrency — shed, don't queue
    :busy

  {:error, {:invalid_input, reason}} ->
    {:bad_request, reason}

  {:error, {:backend_error, error}} ->
    Logger.error(Exception.format(:error, error, error.stacktrace))
    :error
end
```

`MLServe.Error.retryable?/1` separates transient failures from permanent ones:

```elixir
if MLServe.Error.retryable?(reason), do: retry(), else: give_up()
```

`:timeout`, `:overloaded` and `:model_not_ready` are retryable. `:model_not_found`,
`{:invalid_input, _}` and a backend that raised are not — retrying re-runs the same failure.

### Backend exceptions

A backend that raises produces a `MLServe.BackendError` carrying the original exception **and
stacktrace**:

```elixir
{:error, {:backend_error, error}} = MLServe.predict(:fraud_detection, weird_input)

error.backend      #=> MyApp.Backends.ONNX
error.callback     #=> {:predict, 2}
error.kind         #=> :error
error.reason       #=> %ArgumentError{message: "expected shape {1, 3}, got {1, 2}"}
error.stacktrace   #=> [{MyApp.Backends.ONNX, :predict, 2, [...]}, ...]
```

Catching is only ever used to attach context, never to hide the failure — the same event is
reported on `[:ml_serve, :prediction, :exception]` at the same moment.

## Checking availability

```elixir
MLServe.ready?(:fraud_detection)              #=> true
MLServe.ready?({:fraud_detection, "2.1.0"})   #=> a specific version
MLServe.ready?()                              #=> every registered model
MLServe.await_ready(:fraud_detection, 5_000)  #=> :ok | {:error, :timeout}
```

`ready?/0` is what a Kubernetes readiness probe should call: it is false while models load, so
traffic is not routed to an instance that cannot serve it yet.
