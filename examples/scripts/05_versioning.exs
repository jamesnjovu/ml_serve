# Model versioning — shipping a new model without a deploy and without downtime.
#
#     elixir examples/scripts/05_versioning.exs
#
# Covers: running two versions side by side, pinning a version, canary rollout by percentage,
# comparing versions from telemetry, promotion, and draining the old version.

Mix.install([
  # Running from a clone of the repo. Against the published package this is:
  #   {:ml_serve, "~> 0.1.0"}
  {:ml_serve, path: Path.expand("../..", __DIR__)}
])

section = fn title ->
  IO.puts("\n── " <> title <> " " <> String.duplicate("─", max(0, 62 - String.length(title))))
end

defmodule Recommender do
  @moduledoc "Two generations of the same model, distinguishable by what they return."
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(config), do: {:ok, Keyword.fetch!(config, :generation)}

  @impl true
  def predict(generation, user_id) do
    {:ok, %{user: user_id, picks: Enum.map(1..3, &(&1 * generation)), generation: generation}}
  end
end

defmodule BrokenModel do
  @moduledoc "A candidate whose weights will not load, for the failure path below."
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(_config), do: {:error, :weights_file_corrupt}

  @impl true
  def predict(_state, _input), do: {:ok, :never_reached}
end

defmodule Rollout do
  @moduledoc """
  Tallies served requests by version, straight from prediction telemetry.

  This is the whole reason `:version` and `:canary?` are in the metadata: comparing a candidate
  against the incumbent needs no extra instrumentation on your side, and works identically
  whether your metrics backend is this ETS table or Datadog.
  """

  def start do
    :ets.new(:rollout, [:public, :named_table])

    :telemetry.attach(
      "rollout",
      [:ml_serve, :prediction, :stop],
      &__MODULE__.handle_event/4,
      nil
    )
  end

  def handle_event(_event, _measurements, %{version: version, canary?: canary?}, _config) do
    :ets.update_counter(:rollout, {version, canary?}, {2, 1}, {{version, canary?}, 0})
    :ok
  end

  def tally do
    :rollout
    |> :ets.tab2list()
    |> Enum.sort()
  end

  def reset, do: :ets.delete_all_objects(:rollout)
end

Rollout.start()

section.("Two versions, side by side")

# Models are keyed by {name, version}. Loading a second version does not disturb the first: it
# gets its own supervision subtree, its own workers and its own counters.
{:ok, _} =
  MLServe.load_model(:recommender,
    backend: Recommender,
    version: "1.0.0",
    config: [generation: 1]
  )

{:ok, _} =
  MLServe.load_model(:recommender,
    backend: Recommender,
    version: "2.0.0",
    config: [generation: 2]
  )

:ok = MLServe.await_ready(:recommender)

{:ok, versions} = MLServe.versions(:recommender)
IO.puts("    loaded versions   #{inspect(versions)}")

{:ok, status} = MLServe.model_status(:recommender)
IO.puts("    default version   #{status.version}")

# Unpinned traffic follows the default version, which is the first one loaded until you promote.
{:ok, result} = MLServe.predict(:recommender, "user-1")
IO.puts("    unpinned call     generation #{result.generation}")

# Pinning bypasses default routing and any canary entirely. Useful for A/B harnesses, for
# replaying a request against a specific version, and for backfills that must stay reproducible.
{:ok, pinned} = MLServe.predict(:recommender, "user-1", version: "2.0.0")
IO.puts("    pinned to 2.0.0   generation #{pinned.generation}")

section.("Canary: 20% of traffic to the candidate")

# Each request rolls independently in the calling process. There is no coordinator and no shared
# counter, so this costs nothing and cannot become a bottleneck.
:ok = MLServe.canary(:recommender, "2.0.0", 20)
Rollout.reset()

for i <- 1..1_000, do: MLServe.predict!(:recommender, "user-#{i}")

for {{version, canary?}, count} <- Rollout.tally() do
  bar = String.duplicate("▪", div(count, 20))
  label = if canary?, do: "#{version} (canary)", else: "#{version} (default)"

  IO.puts(
    "    #{String.pad_trailing(label, 18)} #{String.pad_leading(to_string(count), 4)}  #{bar}"
  )
end

IO.puts("""

    Roughly 20% landed on the candidate. Both rows carry a :version, so error rate and latency
    per version are already in your metrics — that is what makes the promote/abort call a
    decision rather than a guess.\
""")

section.("A candidate that cannot load never black-holes its share")

# A model that fails to load retries with exponential backoff before being marked :failed, with
# the reason preserved. The warnings below are that backoff, and are the point of this section.
{:ok, _} = MLServe.load_model(:recommender, backend: BrokenModel, version: "3.0.0")

# Long enough for the retry budget to be exhausted.
Process.sleep(1_200)

{:ok, broken} = MLServe.model_status(:recommender, version: "3.0.0")
IO.puts("\n    3.0.0 status      #{broken.status} (#{inspect(broken.failure)})")

# Sending it traffic anyway. Routing checks readiness per request, so the canary share falls
# back to the default version instead of erroring — a bad candidate cannot take production down.
:ok = MLServe.canary(:recommender, "3.0.0", 50)
Rollout.reset()

for i <- 1..500, do: MLServe.predict!(:recommender, "user-#{i}")

IO.puts("    canary at 50%     #{inspect(Rollout.tally())}")
IO.puts("    ...all 500 served by a ready version, none by 3.0.0")

:ok = MLServe.clear_canary(:recommender)

section.("Promote")

Rollout.reset()

# A single ETS write. In-flight requests finish on whatever version they started on; the next
# request routes to the new default. No restart, no dropped connection, no deploy.
:ok = MLServe.promote(:recommender, "2.0.0")

for i <- 1..200, do: MLServe.predict!(:recommender, "user-#{i}")

IO.puts("    after promote     #{inspect(Rollout.tally())}")
IO.puts("    canary cleared    #{inspect(elem(MLServe.model_status(:recommender), 1).canary)}")

section.("Drain the old version")

{:ok, old} = MLServe.model_status(:recommender, version: "1.0.0")
IO.puts("    1.0.0 served      #{old.requests} requests, #{old.errors} errors")

# Waits up to :drain_timeout for in-flight requests before tearing the subtree down. The
# [:ml_serve, :model, :unload] event reports how many were still running when the wait expired —
# 0 is a clean drain.
:ok = MLServe.unload_model(:recommender, version: "1.0.0")

{:ok, remaining} = MLServe.versions(:recommender)
IO.puts("    remaining         #{inspect(remaining)}")

MLServe.unload_model(:recommender, version: "3.0.0")
MLServe.unload_model(:recommender, version: "2.0.0")
