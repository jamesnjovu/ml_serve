# Oban Integration

Oban is **not** a dependency of MLServe. It does not need to be: an Oban worker is a process, and
`MLServe.predict/3` runs in the calling process. The interesting part is not wiring them together
— it is getting retry semantics right, and knowing when a job is the wrong tool.

## When to use a job

| Situation | Use |
| --- | --- |
| Scoring in a request path | `MLServe.predict/3` directly. A job adds latency and a round-trip to Postgres. |
| Inference that takes seconds | An Oban job |
| Scoring a table, a backfill, a nightly run | An Oban job with `batch_predict/3` |
| Work that must survive a deploy or crash | An Oban job — MLServe has no durable queue, deliberately |
| Smoothing bursty load | `max_concurrency:` and dynamic batching, **not** a job queue |

MLServe's `:max_concurrency` sheds load; Oban's queue concurrency *defers* it durably. Those are
different tools for different problems.

## A worker

```elixir
defmodule MyApp.Workers.FraudScorer do
  use Oban.Worker, queue: :inference, max_attempts: 5

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"transaction_id" => id}}) do
    transaction = MyApp.Repo.get!(MyApp.Transaction, id)

    case MLServe.predict(:fraud_detection, features(transaction)) do
      {:ok, result} ->
        MyApp.Transactions.record_score(transaction, result)
        :ok

      {:error, reason} ->
        handle(reason)
    end
  end

  # This is the part worth getting right. Retrying a permanent failure burns attempts and fills
  # your dead-letter queue with jobs that can never succeed; discarding a transient one throws
  # away work that would have succeeded a second later.
  defp handle(reason) do
    error = MLServe.Error.wrap(reason)

    if MLServe.Error.retryable?(error) do
      {:snooze, snooze_for(error)}
    else
      {:discard, Exception.message(error)}
    end
  end

  defp snooze_for(%MLServe.Error{type: :model_not_ready}), do: 5   # still loading
  defp snooze_for(%MLServe.Error{type: :overloaded}), do: 10       # back off
  defp snooze_for(_), do: 2                                        # :timeout

  defp features(transaction) do
    %{
      amount: transaction.amount,
      transaction_count_24h: transaction.count_24h,
      failed_transactions_24h: transaction.failed_24h
    }
  end
end
```

`MLServe.Error.retryable?/1` exists for exactly this decision:

| Retryable | Not retryable |
| --- | --- |
| `:timeout` | `:model_not_found` |
| `:overloaded` | `{:invalid_input, _}` |
| `:model_not_ready` | `{:batch_too_large, _}` |
| | `{:backend_error, _}` — the backend raised; retrying re-runs the same bug |

Note `:snooze` rather than `{:error, reason}` for transient failures: a snooze does not consume an
attempt, so a model that is briefly overloaded does not exhaust `max_attempts` and land in the
dead-letter queue for a condition that resolved in seconds.

## Batch jobs

For a backfill, one job per row means one backend round-trip per row. Chunk instead:

```elixir
defmodule MyApp.Workers.BatchScorer do
  use Oban.Worker, queue: :inference, max_attempts: 3

  @chunk_size 256

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"transaction_ids" => ids}}) do
    transactions = MyApp.Repo.all(from t in MyApp.Transaction, where: t.id in ^ids)
    inputs = Enum.map(transactions, &features/1)

    case MLServe.batch_predict(:fraud_detection, inputs, timeout: :timer.minutes(2)) do
      {:ok, results} ->
        transactions
        |> Enum.zip(results)
        |> MyApp.Transactions.record_scores()

      {:error, reason} ->
        if MLServe.Error.retryable?(reason), do: {:snooze, 30}, else: {:discard, inspect(reason)}
    end
  end

  # Chunk to the model's max_batch_size, not to a number you like the look of.
  def enqueue_all(ids) do
    ids
    |> Enum.chunk_every(@chunk_size)
    |> Enum.map(&new(%{transaction_ids: &1}))
    |> Oban.insert_all()
  end
end
```

Two things to keep aligned:

- **`@chunk_size` must not exceed the model's `:max_batch_size`**, or every job fails immediately
  with `{:batch_too_large, max}`.
- **The job's `:timeout` must exceed the batch's.** A 256-row batch is not a 5-second operation;
  the default per-request timeout will not cover it.

## Queue concurrency and worker pools

```elixir
config :my_app, Oban,
  queues: [inference: 4]              # at most 4 jobs scoring concurrently

config :ml_serve,
  models: [fraud_detection: [workers: 4]]
```

Keep Oban's queue concurrency at or below the model's `workers`. Going higher just queues work
inside MLServe that Oban has already queued durably in Postgres — two queues for one bottleneck,
where the outer one has crash recovery and the inner one does not.

If the model uses `:shared` concurrency (no workers), bound it with `:max_concurrency` instead and
match the queue to that.

## Waiting for models at startup

Oban may start draining its queue before models have finished loading. That is fine — jobs snooze
— but a cleaner option is to have jobs check first:

```elixir
@impl Oban.Worker
def perform(%Oban.Job{args: args}) do
  case MLServe.await_ready(:fraud_detection, 5_000) do
    :ok -> do_perform(args)
    {:error, :timeout} -> {:snooze, 15}
  end
end
```

Or pause the queue until models are ready:

```elixir
defmodule MyApp.InferenceGate do
  use Task, restart: :transient

  def start_link(_), do: Task.start_link(__MODULE__, :run, [])

  def run do
    Oban.pause_queue(queue: :inference)

    case MLServe.await_ready(:fraud_detection, :timer.minutes(5)) do
      :ok -> Oban.resume_queue(queue: :inference)
      {:error, :timeout} -> raise "fraud_detection never became ready"
    end
  end
end
```

## Unique jobs

Scoring the same row twice is usually waste rather than a bug, but it is easy to avoid:

```elixir
use Oban.Worker,
  queue: :inference,
  unique: [period: 300, fields: [:worker, :args]]
```

## Telemetry across both

Oban and MLServe both emit telemetry, and joining them tells you where time actually goes:

```elixir
:telemetry.attach("inference-job-timing", [:oban, :job, :stop], fn _e, measurements, meta, _c ->
  if meta.queue == "inference" do
    :telemetry.execute([:my_app, :inference, :job], %{duration: measurements.duration}, %{
      worker: meta.worker,
      attempt: meta.job.attempt
    })
  end
end, nil)
```

Compare `[:oban, :job, :stop]` duration against `[:ml_serve, :prediction, :stop]` duration for the
same work: a large gap is time spent in `Repo` calls, not in the model, and the fix is a database
one.

## Testing

```elixir
defmodule MyApp.Workers.FraudScorerTest do
  use MyApp.DataCase, async: true
  use Oban.Testing, repo: MyApp.Repo

  test "records a score" do
    transaction = insert(:transaction, amount: 1500.50)

    assert :ok = perform_job(MyApp.Workers.FraudScorer, %{transaction_id: transaction.id})
    assert MyApp.Repo.reload(transaction).fraud_score
  end

  test "snoozes when the model is not ready" do
    name = :"unready_#{System.unique_integer([:positive])}"
    # A model that never loads, so predictions return :model_not_ready.
    MLServe.load_model(name, backend: MyApp.NeverLoadsBackend)
    on_exit(fn -> MLServe.unload_model(name) end)

    assert {:snooze, _} = MyApp.Workers.FraudScorer.perform(%Oban.Job{args: %{...}})
  end
end
```

With `MLServe.Backend.Static` configured in `config/test.exs`, the happy path needs no model file
and no ML runtime.
