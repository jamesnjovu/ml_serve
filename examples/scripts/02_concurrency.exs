# Concurrency — where inference actually runs.
#
#     elixir examples/scripts/02_concurrency.exs
#
# Covers: :shared vs :exclusive execution, worker pools, worker selection, and the admission
# limit that turns overload into a fast {:error, :overloaded} instead of a queue that grows
# until the node dies.

Mix.install([
  # Running from a clone of the repo. Against the published package this is:
  #   {:ml_serve, "~> 0.1.0"}
  {:ml_serve, path: Path.expand("../..", __DIR__)}
])

section = fn title ->
  IO.puts("\n── " <> title <> " " <> String.duplicate("─", max(0, 62 - String.length(title))))
end

defmodule WhereAmI do
  @moduledoc """
  Reports the process that ran the inference, which is the entire point of this example.

  The backend declares nothing about concurrency, so both models below set `:concurrency`
  explicitly. Real backends declare it once in `capabilities/0` — the option exists so you can
  override a backend you do not own.
  """
  @behaviour MLServe.Model

  @impl true
  def load(config), do: {:ok, Keyword.get(config, :delay, 0)}

  @impl true
  def predict(delay, _input) do
    if delay > 0, do: Process.sleep(delay)
    {:ok, self()}
  end
end

# `:shared` — inference runs in the calling process, with state read from :persistent_term. No
# worker processes exist, nothing is copied between mailboxes, and nothing serialises traffic.
# This is the right shape for an Nx.Serving, Bumblebee, a pure function or an HTTP call.
{:ok, _} =
  MLServe.load_model(:shared_model, backend: WhereAmI, concurrency: :shared)

# `:exclusive` — inference runs in a supervised worker pool. Necessary when the backend is not
# safe to call concurrently: an ONNX session, a port to a Python process, anything with a lock.
{:ok, _} =
  MLServe.load_model(:pooled_model, backend: WhereAmI, concurrency: :exclusive, workers: 4)

:ok = MLServe.await_ready()

section.("Who runs the inference?")

caller = self()
{:ok, shared_pid} = MLServe.predict(:shared_model, :anything)
{:ok, pooled_pid} = MLServe.predict(:pooled_model, :anything)

IO.puts("  caller                #{inspect(caller)}")
IO.puts("  :shared ran in        #{inspect(shared_pid)}   same process? #{shared_pid == caller}")
IO.puts("  :exclusive ran in     #{inspect(pooled_pid)}   same process? #{pooled_pid == caller}")

section.("Worker counts")

for model <- [:shared_model, :pooled_model] do
  {:ok, status} = MLServe.model_status(model)

  IO.puts(
    "  #{String.pad_trailing(to_string(model), 16)} #{status.concurrency}\tworkers: #{status.workers}"
  )
end

IO.puts(
  "\n  A :shared model starts zero workers. There is no pool to size and no queue to wait in."
)

section.("Spreading load across the pool")

# 200 concurrent predictions against a 4-worker pool. Round-robin selection means every worker
# takes a roughly equal share.
distribution =
  1..200
  |> Task.async_stream(fn _ -> MLServe.predict!(:pooled_model, :work) end, max_concurrency: 50)
  |> Enum.frequencies_by(fn {:ok, pid} -> pid end)

IO.puts("  #{map_size(distribution)} distinct workers served 200 requests:")

for {pid, count} <- Enum.sort_by(distribution, &elem(&1, 1), :desc) do
  IO.puts("    #{inspect(pid)}  #{String.duplicate("▪", div(count, 2))} #{count}")
end

# :least_loaded picks the worker with the fewest requests in flight, which is what you want when
# inference times vary widely. :random skips the counter read entirely.
IO.puts("\n  (:selection accepts :round_robin — the default — plus :least_loaded and :random)")

section.("Shedding load instead of queueing it")

# Without a limit, a pool under sustained overload grows an unbounded queue and every caller
# waits longer than the last. :max_concurrency rejects the excess immediately, so callers can
# fall back to a cached answer or a simpler model rather than timing out.
{:ok, _} =
  MLServe.load_model(:limited_model,
    backend: WhereAmI,
    concurrency: :exclusive,
    workers: 2,
    max_concurrency: 2,
    config: [delay: 150]
  )

:ok = MLServe.await_ready(:limited_model)

outcomes =
  1..10
  |> Task.async_stream(fn _ -> MLServe.predict(:limited_model, :work) end,
    max_concurrency: 10,
    timeout: 5_000
  )
  |> Enum.frequencies_by(fn
    {:ok, {:ok, _pid}} -> :served
    {:ok, {:error, reason}} -> reason
  end)

IO.inspect(outcomes, label: "  10 concurrent requests against max_concurrency: 2")

IO.puts("""

  Admission control happens in the *caller*, before a worker is involved, so a rejected request
  costs one ETS read. MLServe.Error.retryable?/1 reports true for :overloaded — it is a
  back-pressure signal, not a failure.\
""")

for model <- [:shared_model, :pooled_model, :limited_model], do: MLServe.unload_model(model)
