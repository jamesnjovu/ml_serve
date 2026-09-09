defmodule MLServe.Cache do
  @moduledoc """
  Optional TTL cache for inference results.

  **Off by default, and deliberately so.** Caching a prediction is only correct when the same
  input must produce the same output — which is false the moment a model reads a clock, a random
  seed, or mutable feature state. Silently caching by default would turn that into a subtle
  correctness bug, so `MLServe.predict/3` caches only when you ask:

      MLServe.predict(:fraud_detection, features, cache: true)

  or per model:

      config :ml_serve,
        models: [fraud_detection: [cache: [enabled: true, ttl: :timer.minutes(1)]]]

  ## Design

  A single `:public` ETS table with `write_concurrency: true`. Readers and writers work directly
  in the calling process — the GenServer exists only to own the table and run the sweeper, and is
  never in the request path.

  Expiry is both lazy (checked on read) and swept (a periodic `:ets.select_delete`). Lazy alone
  leaks memory for keys never read again; sweeping alone lets a reader see a stale entry between
  sweeps.

  ## Keys

  The default key is `:erlang.term_to_binary(input, [:deterministic])` hashed with SHA-256. The
  `:deterministic` flag matters: without it, large maps serialise in internal-hash order and two
  equal inputs can produce different binaries, silently halving the hit rate.

  For large tensor inputs, hashing the whole term is more expensive than the inference you are
  trying to skip. Pass a cheap key instead:

      MLServe.predict(:fraud, features, cache: true, cache_key: features.account_id)

  ## What is not cached

  Errors are never cached — a transient backend failure must not be pinned for the TTL.

  ## Eviction

  TTL plus a `:max_size` bound, enforced at sweep time. There is no LRU: tracking recency needs a
  write on every read, which would put a serialised write on the hot path to save memory that a
  TTL already bounds. If you need LRU, use Cachex and pass results through `:postprocess`.
  """

  use GenServer

  require Logger

  alias MLServe.Telemetry

  @table :ml_serve_cache

  # Client API

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Fetches a cached result, returning `:miss` when absent or expired.

  Emits `[:ml_serve, :cache, :hit]` or `[:ml_serve, :cache, :miss]`.
  """
  @spec fetch(term(), map()) :: {:ok, term()} | :miss
  def fetch(key, metadata \\ %{}) do
    case :ets.lookup(@table, key) do
      [{^key, value, expires_at}] ->
        if expires_at > System.monotonic_time(:millisecond) do
          Telemetry.cache(:hit, metadata)
          {:ok, value}
        else
          :ets.delete(@table, key)
          Telemetry.cache(:miss, metadata)
          :miss
        end

      [] ->
        Telemetry.cache(:miss, metadata)
        :miss
    end
  rescue
    ArgumentError -> :miss
  end

  @doc """
  Stores a result under `key` for `ttl` milliseconds.
  """
  @spec put(term(), term(), pos_integer()) :: :ok
  def put(key, value, ttl) do
    expires_at = System.monotonic_time(:millisecond) + ttl
    :ets.insert(@table, {key, value, expires_at})
    :ok
  rescue
    ArgumentError -> :ok
  end

  @doc """
  Builds the cache key for a prediction.

  ## Parameters

    - `name`: model name
    - `version`: the version serving the request
    - `input`: the input term, or an explicit `:cache_key` value

  ## Examples

      iex> key = MLServe.Cache.key(:fraud, "1.0.0", %{amount: 100})
      iex> match?({:fraud, "1.0.0", _digest}, key)
      true

      iex> MLServe.Cache.key(:fraud, "1.0.0", %{b: 1, a: 2}) ==
      ...>   MLServe.Cache.key(:fraud, "1.0.0", %{a: 2, b: 1})
      true
  """
  @spec key(atom(), String.t(), term()) :: {atom(), String.t(), binary()}
  def key(name, version, input) do
    digest = :crypto.hash(:sha256, :erlang.term_to_binary(input, [:deterministic]))
    {name, version, digest}
  end

  @doc "Removes every cached entry for a model, at every version."
  @spec invalidate(atom()) :: :ok
  def invalidate(name) do
    :ets.match_delete(@table, {{name, :_, :_}, :_, :_})
    :ok
  rescue
    ArgumentError -> :ok
  end

  @doc "Removes every cached entry for one model version."
  @spec invalidate(atom(), String.t()) :: :ok
  def invalidate(name, version) do
    :ets.match_delete(@table, {{name, version, :_}, :_, :_})
    :ok
  rescue
    ArgumentError -> :ok
  end

  @doc "Empties the cache."
  @spec clear() :: :ok
  def clear do
    :ets.delete_all_objects(@table)
    :ok
  rescue
    ArgumentError -> :ok
  end

  @doc "Number of entries currently held, including any not yet swept."
  @spec size() :: non_neg_integer()
  def size do
    :ets.info(@table, :size) || 0
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    table =
      :ets.new(@table, [
        :set,
        :public,
        :named_table,
        read_concurrency: true,
        write_concurrency: true,
        decentralized_counters: true
      ])

    config = MLServe.Config.cache()
    schedule_sweep(config.sweep_interval)

    {:ok, %{table: table, config: config}}
  end

  @impl true
  def handle_info(:sweep, state) do
    now = System.monotonic_time(:millisecond)
    expired = :ets.select_delete(@table, [{{:_, :_, :"$1"}, [{:"=<", :"$1", now}], [true]}])

    over = size() - state.config.max_size

    if over > 0 do
      trim(over)
    end

    if expired > 0 or over > 0 do
      Logger.debug(fn ->
        "[ml_serve] cache sweep removed #{expired} expired and #{max(over, 0)} overflow entries"
      end)
    end

    schedule_sweep(state.config.sweep_interval)
    {:noreply, state}
  end

  def handle_info(_message, state), do: {:noreply, state}

  # Private Functions

  defp schedule_sweep(interval), do: Process.send_after(self(), :sweep, interval)

  @doc false
  # Over the size bound, drop the soonest-to-expire entries first. Not LRU, but it evicts the
  # entries with the least remaining value and needs no per-read bookkeeping on the hot path.
  @spec trim(non_neg_integer()) :: :ok
  def trim(count) when count > 0 do
    @table
    |> :ets.tab2list()
    |> Enum.sort_by(fn {_key, _value, expires_at} -> expires_at end)
    |> Enum.take(count)
    |> Enum.each(fn {key, _value, _expires_at} -> :ets.delete(@table, key) end)
  end

  def trim(_count), do: :ok
end
