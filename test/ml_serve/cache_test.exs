defmodule MLServe.CacheTest do
  use MLServe.Case, async: true

  alias MLServe.Cache
  alias MLServe.Test.Backends

  describe "opt-in behaviour" do
    test "results are not cached by default" do
      name = load!(backend: Backends.Nondeterministic)

      assert {:ok, first} = MLServe.predict(name, :same)
      assert {:ok, second} = MLServe.predict(name, :same)

      refute first == second
    end

    test "cache: true caches by input" do
      name = load!(backend: Backends.Nondeterministic)

      assert {:ok, first} = MLServe.predict(name, :same, cache: true)
      assert {:ok, second} = MLServe.predict(name, :same, cache: true)

      assert first == second
    end

    test "different inputs get different entries" do
      name = load!(backend: Backends.Nondeterministic)

      {:ok, a} = MLServe.predict(name, :one, cache: true)
      {:ok, b} = MLServe.predict(name, :two, cache: true)

      refute a == b
    end

    test "a per-model cache config enables caching without a per-call option" do
      name = load!(backend: Backends.Nondeterministic, cache: [enabled: true, ttl: 60_000])

      {:ok, first} = MLServe.predict(name, :same)
      {:ok, second} = MLServe.predict(name, :same)

      assert first == second
    end

    test "cache: false overrides a per-model cache config" do
      name = load!(backend: Backends.Nondeterministic, cache: [enabled: true, ttl: 60_000])

      {:ok, first} = MLServe.predict(name, :same)
      {:ok, second} = MLServe.predict(name, :same, cache: false)

      refute first == second
    end

    test "the backend is not invoked on a hit" do
      tok = token()
      name = load!(backend: Backends.Counting, workers: 1, config: [token: tok])

      MLServe.predict(name, :x, cache: true)
      MLServe.predict(name, :x, cache: true)
      MLServe.predict(name, :x, cache: true)

      assert Backends.count(tok, :predict) == 1
    end
  end

  describe "TTL" do
    test "an entry expires after its TTL" do
      name = load!(backend: Backends.Nondeterministic)

      {:ok, first} = MLServe.predict(name, :same, cache: true, cache_ttl: 20)
      Process.sleep(40)
      {:ok, second} = MLServe.predict(name, :same, cache: true, cache_ttl: 20)

      refute first == second
    end

    test "an integer :cache option is treated as a TTL" do
      name = load!(backend: Backends.Nondeterministic)

      {:ok, first} = MLServe.predict(name, :same, cache: 5_000)
      {:ok, second} = MLServe.predict(name, :same, cache: 5_000)

      assert first == second
    end
  end

  describe "cache keys" do
    test "maps with the same pairs in a different order share a key" do
      # term_to_binary/2 with :deterministic — without it, large maps serialise in internal-hash
      # order and two equal inputs would silently miss each other.
      assert Cache.key(:m, "1.0.0", %{a: 1, b: 2}) == Cache.key(:m, "1.0.0", %{b: 2, a: 1})
    end

    test "different versions of a model do not share cache entries" do
      refute Cache.key(:m, "1.0.0", :x) == Cache.key(:m, "2.0.0", :x)
    end

    test "an explicit cache_key overrides the input" do
      name = load!(backend: Backends.Nondeterministic)

      {:ok, first} = MLServe.predict(name, :input_a, cache: true, cache_key: :shared)
      {:ok, second} = MLServe.predict(name, :input_b, cache: true, cache_key: :shared)

      assert first == second
    end
  end

  describe "errors" do
    test "errors are never cached" do
      tok = token()

      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [
            predict: fn _n ->
              Backends.record(tok, :predict)
              {:error, :transient}
            end
          ]
        )

      assert MLServe.predict(name, :x, cache: true) == {:error, :transient}
      assert MLServe.predict(name, :x, cache: true) == {:error, :transient}

      # A transient failure must not be pinned for the whole TTL.
      assert Backends.count(tok, :predict) == 2
    end
  end

  describe "telemetry" do
    test "emits hit and miss events" do
      ref = attach_telemetry([[:ml_serve, :cache, :hit], [:ml_serve, :cache, :miss]])
      name = load!(backend: Backends.Echo)

      MLServe.predict(name, :x, cache: true)
      assert {%{count: 1}, _} = assert_telemetry(ref, [:ml_serve, :cache, :miss], name)

      MLServe.predict(name, :x, cache: true)
      assert {%{count: 1}, _} = assert_telemetry(ref, [:ml_serve, :cache, :hit], name)
    end

    test "a cache hit is flagged in the prediction stop event" do
      ref = attach_telemetry([[:ml_serve, :prediction, :stop]])
      name = load!(backend: Backends.Echo)

      MLServe.predict(name, :x, cache: true)
      assert {_, %{cached?: false}} = assert_telemetry(ref, [:ml_serve, :prediction, :stop], name)

      MLServe.predict(name, :x, cache: true)
      assert {_, %{cached?: true}} = assert_telemetry(ref, [:ml_serve, :prediction, :stop], name)
    end
  end

  describe "invalidation" do
    test "unloading a model version clears its cached entries" do
      name = load!(backend: Backends.Nondeterministic)

      {:ok, first} = MLServe.predict(name, :same, cache: true, cache_ttl: 60_000)
      :ok = MLServe.unload_model(name)

      reloaded = load!([backend: Backends.Nondeterministic], name)
      {:ok, second} = MLServe.predict(reloaded, :same, cache: true, cache_ttl: 60_000)

      refute first == second
    end
  end

  describe "direct API" do
    test "put and fetch round-trip" do
      key = {:direct, "1.0.0", :erlang.unique_integer()}

      assert Cache.fetch(key) == :miss
      assert Cache.put(key, :value, 1_000) == :ok
      assert Cache.fetch(key) == {:ok, :value}
    end

    test "fetch reports a miss for an expired entry" do
      key = {:direct, "1.0.0", :erlang.unique_integer()}
      Cache.put(key, :value, 1)
      Process.sleep(20)

      assert Cache.fetch(key) == :miss
    end
  end

  describe "maintenance" do
    test "size/0 reflects stored entries" do
      before = Cache.size()
      key = {:size_check, "1.0.0", :erlang.unique_integer()}
      Cache.put(key, :value, 60_000)

      assert Cache.size() >= before + 1
    end

    test "invalidate/1 clears every version of a model" do
      name = unique_name()
      Cache.put(Cache.key(name, "1.0.0", :a), :v1, 60_000)
      Cache.put(Cache.key(name, "2.0.0", :a), :v2, 60_000)

      assert Cache.invalidate(name) == :ok

      assert Cache.fetch(Cache.key(name, "1.0.0", :a)) == :miss
      assert Cache.fetch(Cache.key(name, "2.0.0", :a)) == :miss
    end

    test "invalidate/2 clears only the named version" do
      name = unique_name()
      Cache.put(Cache.key(name, "1.0.0", :a), :v1, 60_000)
      Cache.put(Cache.key(name, "2.0.0", :a), :v2, 60_000)

      assert Cache.invalidate(name, "1.0.0") == :ok

      assert Cache.fetch(Cache.key(name, "1.0.0", :a)) == :miss
      assert Cache.fetch(Cache.key(name, "2.0.0", :a)) == {:ok, :v2}
    end

    test "the sweeper removes expired entries" do
      name = unique_name()
      key = Cache.key(name, "1.0.0", :swept)
      Cache.put(key, :value, 1)
      Process.sleep(20)

      # Drive the sweep directly rather than waiting for the interval: this asserts the select
      # pattern is right, which is the part that could silently stop matching.
      send(Process.whereis(MLServe.Cache), :sweep)
      _ = :sys.get_state(MLServe.Cache)

      # :ets.lookup rather than fetch/2, so a lazy expiry cannot mask a broken sweeper.
      assert :ets.lookup(:ml_serve_cache, key) == []
    end

    test "the sweeper keeps live entries" do
      name = unique_name()
      key = Cache.key(name, "1.0.0", :kept)
      Cache.put(key, :value, 60_000)

      send(Process.whereis(MLServe.Cache), :sweep)
      _ = :sys.get_state(MLServe.Cache)

      assert Cache.fetch(key) == {:ok, :value}
    end

    test "unknown messages do not crash the cache" do
      pid = Process.whereis(MLServe.Cache)
      send(pid, :something_unexpected)

      assert :sys.get_state(pid)
      assert Process.alive?(pid)
    end
  end
end

defmodule MLServe.CacheGlobalTest do
  # Not async: clear/0 and trim/1 act on the whole table, not one model, so they cannot run
  # alongside the async cache tests — trim/1 would evict another test's entry. ExUnit runs sync
  # cases only after every async one has finished.
  use MLServe.Case, async: false

  alias MLServe.Cache

  setup do
    Cache.clear()
    :ok
  end

  test "clear/0 empties the cache" do
    key = Cache.key(unique_name(), "1.0.0", :cleared)
    Cache.put(key, :value, 60_000)
    assert Cache.size() > 0

    assert Cache.clear() == :ok

    assert Cache.fetch(key) == :miss
    assert Cache.size() == 0
  end

  test "trim/1 evicts the soonest-to-expire entries first" do
    name = unique_name()
    soon = Cache.key(name, "1.0.0", :soon)
    later = Cache.key(name, "1.0.0", :later)
    latest = Cache.key(name, "1.0.0", :latest)

    Cache.put(soon, :a, 1_000)
    Cache.put(later, :b, 60_000)
    Cache.put(latest, :c, 600_000)

    # Trimming picks by remaining TTL, not insertion order: the entries with the least value
    # left go first. Not LRU, but it needs no bookkeeping on the read path.
    Cache.trim(1)

    assert Cache.fetch(soon) == :miss
    assert Cache.fetch(later) == {:ok, :b}
    assert Cache.fetch(latest) == {:ok, :c}
  end

  test "trim/1 with a non-positive count is a no-op" do
    key = Cache.key(unique_name(), "1.0.0", :kept)
    Cache.put(key, :value, 60_000)

    assert Cache.trim(0) == :ok
    assert Cache.trim(-5) == :ok
    assert Cache.fetch(key) == {:ok, :value}
  end
end
