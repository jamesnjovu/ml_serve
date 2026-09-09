# Caching — skipping inference you have already done.
#
#     elixir examples/scripts/04_caching.exs
#
# Covers: why the cache is off by default, enabling it per call and per model, cheap cache keys
# for expensive inputs, TTL expiry, why errors are never cached, and invalidation.

Mix.install([
  # Running from a clone of the repo. Against the published package this is:
  #   {:ml_serve, "~> 0.1.0"}
  {:ml_serve, path: Path.expand("../..", __DIR__)}
])

section = fn title ->
  IO.puts("\n── " <> title <> " " <> String.duplicate("─", max(0, 62 - String.length(title))))
end

defmodule EmbeddingModel do
  @moduledoc """
  Stands in for an expensive, deterministic model — the only kind it is safe to cache.

  Counts its invocations so a cache hit is visible as work that did not happen.
  """
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(_config), do: {:ok, :counters.new(1, [])}

  @impl true
  def predict(_counters, %{fail: true}), do: {:error, :upstream_unavailable}

  def predict(counters, %{text: text}) do
    :counters.add(counters, 1, 1)
    Process.sleep(20)
    {:ok, %{embedding: :erlang.phash2(text, 1_000) / 1_000, text: text}}
  end

  def calls(counters), do: :counters.get(counters, 1)
end

defmodule CacheTally do
  @moduledoc "Counts [:ml_serve, :cache, :hit] and [:ml_serve, :cache, :miss]."

  def start do
    :ets.new(:cache_tally, [:public, :named_table])

    :telemetry.attach_many(
      "cache-events",
      [[:ml_serve, :cache, :hit], [:ml_serve, :cache, :miss]],
      &__MODULE__.handle_event/4,
      nil
    )
  end

  def handle_event([:ml_serve, :cache, outcome], _measurements, _metadata, _config) do
    :ets.update_counter(:cache_tally, outcome, {2, 1}, {outcome, 0})
    :ok
  end

  def get(key) do
    case :ets.lookup(:cache_tally, key) do
      [{^key, value}] -> value
      [] -> 0
    end
  end

  def reset, do: :ets.delete_all_objects(:cache_tally)
end

CacheTally.start()

{:ok, _} = MLServe.load_model(:embeddings, backend: EmbeddingModel)
:ok = MLServe.await_ready(:embeddings)

input = %{text: "the quick brown fox"}

time = fn fun ->
  {us, result} = :timer.tc(fun)
  {Float.round(us / 1_000, 1), result}
end

section.("Off by default")

# Caching a prediction is only correct when the same input must produce the same output — false
# the moment a model reads a clock, a random seed or mutable feature state. Defaulting to on
# would turn that into a silent correctness bug, so you have to ask.
{first_ms, _} = time.(fn -> MLServe.predict!(:embeddings, input) end)
{second_ms, _} = time.(fn -> MLServe.predict!(:embeddings, input) end)

IO.puts("""
    call 1    #{first_ms}ms
    call 2    #{second_ms}ms      the backend ran twice; nothing was remembered\
""")

section.("Enabled per call")

CacheTally.reset()

{miss_ms, _} = time.(fn -> MLServe.predict!(:embeddings, input, cache: true) end)
{hit_ms, _} = time.(fn -> MLServe.predict!(:embeddings, input, cache: true) end)
{hit2_ms, _} = time.(fn -> MLServe.predict!(:embeddings, input, cache: true) end)

IO.puts("""
    call 1    #{miss_ms}ms      miss — ran the backend and stored the result
    call 2    #{hit_ms}ms       hit
    call 3    #{hit2_ms}ms       hit

    telemetry: #{CacheTally.get(:hit)} hits, #{CacheTally.get(:miss)} misses\
""")

section.("Cheap keys for expensive inputs")

# The default key is SHA-256 over :erlang.term_to_binary(input, [:deterministic]). For a large
# tensor that hash costs more than the inference it saves. When you already hold something that
# identifies the input, say so — these two calls share a cache entry despite different inputs.
CacheTally.reset()

MLServe.predict!(:embeddings, %{text: "hello"}, cache: true, cache_key: "account:42")

MLServe.predict!(:embeddings, %{text: "completely different"},
  cache: true,
  cache_key: "account:42"
)

IO.puts("""
    two different inputs, one :cache_key → #{CacheTally.get(:hit)} hit, #{CacheTally.get(:miss)} miss

    A cache key is a promise that inputs sharing it produce the same answer. MLServe takes you at
    your word — that is the point, and the risk.\
""")

section.("TTL")

CacheTally.reset()

MLServe.predict!(:embeddings, %{text: "short lived"}, cache: true, cache_ttl: 50)
MLServe.predict!(:embeddings, %{text: "short lived"}, cache: true, cache_ttl: 50)
IO.puts("    immediately        #{CacheTally.get(:hit)} hit")

Process.sleep(80)
MLServe.predict!(:embeddings, %{text: "short lived"}, cache: true, cache_ttl: 50)
IO.puts("    after 80ms         #{CacheTally.get(:miss)} misses — the entry expired")

section.("Errors are never cached")

CacheTally.reset()

for _ <- 1..3, do: MLServe.predict(:embeddings, %{fail: true}, cache: true)

IO.puts("""
    3 failing calls → #{CacheTally.get(:hit)} hits

    A transient backend failure must not be pinned for the whole TTL. Only {:ok, result} is
    stored.\
""")

section.("Per-model configuration and invalidation")

# Rather than passing cache: true at every call site, declare it on the model. Everything else
# is identical — this is the same cache.
{:ok, _} =
  MLServe.load_model(:always_cached,
    backend: EmbeddingModel,
    cache: [enabled: true, ttl: :timer.minutes(5)]
  )

:ok = MLServe.await_ready(:always_cached)
CacheTally.reset()

# Start from an empty table so the entry count below is unambiguous.
:ok = MLServe.Cache.clear()

for _ <- 1..5, do: MLServe.predict!(:always_cached, input)

IO.puts(
  "    5 calls, no :cache option → #{CacheTally.get(:hit)} hits, #{CacheTally.get(:miss)} miss"
)

IO.puts("    entries held               #{MLServe.Cache.size()}")

# Entries are keyed by {name, version, input}, so a new version never serves the old one's
# cached answers. Invalidate explicitly when the *data behind* an unchanged model moves.
:ok = MLServe.Cache.invalidate(:always_cached)
IO.puts("    after invalidate/1        #{MLServe.Cache.size()}")

for model <- [:embeddings, :always_cached], do: MLServe.unload_model(model)
