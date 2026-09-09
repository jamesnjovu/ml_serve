# Dynamic batching — turning many independent callers into one backend call.
#
#     elixir examples/scripts/03_batching.exs
#
# The production shape that matters: many processes each holding *one* input, arriving within
# milliseconds of each other. A GPU that processes 32 rows in barely more time than it processes
# one is wasted by a pool feeding it a row at a time. Batching collects arrivals into a window.

Mix.install([
  # Running from a clone of the repo. Against the published package this is:
  #   {:ml_serve, "~> 0.1.0"}
  {:ml_serve, path: Path.expand("../..", __DIR__)}
])

section = fn title ->
  IO.puts("\n── " <> title <> " " <> String.duplicate("─", max(0, 62 - String.length(title))))
end

defmodule CountingModel do
  @moduledoc """
  Counts how often the backend is actually invoked, and over how many rows.

  `load/1` runs once and its state is shared by every worker, so a `:counters` reference created
  here is a lock-free counter every worker writes to — the same trick a real backend uses to hold
  a single NIF resource across a pool.
  """
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :exclusive, load: :once}

  @impl true
  def load(_config), do: {:ok, :counters.new(1, [:write_concurrency])}

  @impl true
  def predict(counters, input) do
    :counters.add(counters, 1, 1)
    # Stand-in for the fixed per-call overhead every real backend pays whether it is handed one
    # row or a thousand: the NIF boundary, the host-to-device copy, the HTTP round trip.
    Process.sleep(5)
    {:ok, input * 2}
  end

  # The optional callback that makes batching worth anything. One invocation, one fixed cost,
  # N rows. Without it MLServe still batches, but falls back to mapping predict/2 over the list.
  @impl true
  def batch_predict(counters, inputs) do
    :counters.add(counters, 1, 1)
    Process.sleep(5)
    {:ok, Enum.map(inputs, &(&1 * 2))}
  end

  def invocations(counters), do: :counters.get(counters, 1)
end

defmodule FlushTally do
  @moduledoc "Records every batch flush, so the window's behaviour is visible rather than assumed."

  @table :flush_tally

  def start do
    :ets.new(@table, [:public, :named_table])

    :telemetry.attach_many(
      "batch-flushes",
      [[:ml_serve, :batch, :flush]],
      &__MODULE__.handle_event/4,
      nil
    )
  end

  # A named function rather than an anonymous one: :telemetry logs a performance warning for
  # local captures, and in a hot path it means a lookup per event.
  def handle_event(_event, %{size: size}, %{reason: reason}, _config) do
    :ets.update_counter(@table, :flushes, {2, 1}, {:flushes, 0})
    :ets.update_counter(@table, :rows, {2, size}, {:rows, 0})
    :ets.update_counter(@table, reason, {2, 1}, {reason, 0})
    if size > get(:largest), do: :ets.insert(@table, {:largest, size})
    :ok
  end

  def get(key) do
    case :ets.lookup(@table, key) do
      [{^key, value}] -> value
      [] -> 0
    end
  end
end

FlushTally.start()

# Two identical models. The only difference is the window.
{:ok, _} = MLServe.load_model(:unbatched, backend: CountingModel, workers: 4)

# The window closes when :max_size inputs have accumulated (reason: :full) or :timeout ms have
# passed since the *first* input in the batch (reason: :timeout). Timing from the first arrival
# rather than the last bounds the added latency at :timeout for every caller; a sliding window
# timed from the last arrival can starve the earliest caller indefinitely under steady traffic.
{:ok, _} =
  MLServe.load_model(:batched,
    backend: CountingModel,
    workers: 4,
    batching: [max_size: 16, timeout: 10]
  )

:ok = MLServe.await_ready()

requests = 200

run = fn model ->
  {elapsed_us, _} =
    :timer.tc(fn ->
      1..requests
      |> Task.async_stream(&MLServe.predict!(model, &1),
        max_concurrency: requests,
        timeout: 30_000
      )
      |> Stream.run()
    end)

  elapsed_us / 1_000
end

section.("#{requests} concurrent single predictions")

unbatched_ms = run.(:unbatched)
batched_ms = run.(:batched)

IO.puts("""
    without batching    #{Float.round(unbatched_ms, 1)}ms      #{requests} backend invocations
    with batching       #{Float.round(batched_ms, 1)}ms      #{FlushTally.get(:flushes)} backend invocations
    speedup             #{Float.round(unbatched_ms / batched_ms, 1)}×\
""")

section.("How the window actually behaved")

flushes = FlushTally.get(:flushes)
rows = FlushTally.get(:rows)

IO.puts("""
    rows processed        #{rows}
    flushes               #{flushes}
    average batch         #{Float.round(rows / flushes, 1)} rows
    largest batch         #{FlushTally.get(:largest)} rows
    flushed on :full      #{FlushTally.get(:full)}
    flushed on :timeout   #{FlushTally.get(:timeout)}
""")

IO.puts("""
  Note the largest batch exceeds max_size: 16. That is not a bug, and it is the part worth
  understanding: :max_size is the trigger that *opens* a flush, not a cap on what the flush
  carries. A flush takes everything queued at that instant.

  In-flight batches are separately capped at the worker count (4 here). When every worker is
  busy, flushes are held and arrivals keep accumulating — that hold *is* the backpressure. So a
  burst naturally produces larger batches exactly when larger batches pay off most.

  Mostly :full is the healthy signature. Mostly :timeout means you are paying the full window in
  latency for batches that never fill: shorten it, or accept smaller batches.\
""")

section.("One caller with many inputs")

# batch_predict/3 is the other half of the story: a single caller who already holds the whole
# list. It goes straight to the backend's batch_predict/2 in one call, bypassing the window.
{:ok, results} = MLServe.batch_predict(:batched, Enum.to_list(1..8))
IO.inspect(results, label: "  batch_predict(1..8)")

for model <- [:unbatched, :batched], do: MLServe.unload_model(model)
