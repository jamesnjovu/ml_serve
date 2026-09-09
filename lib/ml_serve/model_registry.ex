defmodule MLServe.ModelRegistry do
  @moduledoc """
  The model catalog: which models exist, at which versions, and where traffic goes.

  ## Why this is a GenServer that is never called on the hot path

  `MLServe.predict/3` runs in whatever process is handling the request — a Phoenix controller, an
  Oban worker, a `Task`. If every prediction had to `GenServer.call` this process just to find out
  which worker to talk to, this single process would serialise the entire node's inference
  traffic before any inference happened.

  So the registry owns a `:protected` ETS table with `read_concurrency: true`, and:

    * **Reads** — `route/2`, `status/2`, `list/0` — run `:ets.lookup/2` directly in the calling
      process. Lock-free, and they scale with schedulers.
    * **Writes** — registering, promoting, unregistering — go through the GenServer, which is the
      only process with write access. Serialising writes is free because they happen at load and
      deploy time, not per request.

  ## Table layout

  | Key | Value | Purpose |
  | --- | ----- | ------- |
  | `{:route, name, version}` | `MLServe.Route` | Hot path. Slim by design — see `MLServe.Route`. |
  | `{:model, name, version}` | map | Full detail for `MLServe.model_status/2`. |
  | `{:default, name}` | version | Which version unpinned traffic gets. |
  | `{:canary, name}` | `{version, weight}` | Progressive rollout. |

  Splitting the hot-path route from the full entry keeps the per-prediction ETS copy to a few
  dozen words instead of copying the backend's entire configuration.
  """

  use GenServer

  require Logger

  alias MLServe.ModelSpec
  alias MLServe.Route

  @table :ml_serve_catalog

  # Client API

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Resolves a model name to the route that should serve this request.

  Runs entirely in the calling process. Version selection, in order:

    1. An explicit `version: "2.1.0"` option pins that version exactly.
    2. Otherwise, if a canary is configured, a per-request roll sends `weight`% of traffic to the
       candidate. `:rand.uniform/1` uses the caller's own seed — no shared state, no coordination.
    3. Otherwise the default version.

  Returns `{:error, :model_not_ready}` rather than `:model_not_found` when the model exists but is
  loading, draining or failed, because those are different operational problems.
  """
  @spec route(atom(), keyword()) ::
          {:ok, Route.t()} | {:error, :model_not_found | :model_not_ready}
  def route(name, opts \\ []) do
    case Keyword.get(opts, :version) do
      nil -> resolve_default(name)
      version -> fetch_route(name, version)
    end
  end

  @doc """
  Fetches the route for an exact `{name, version}` pair, ready or not.
  """
  @spec fetch_route(atom(), String.t()) ::
          {:ok, Route.t()} | {:error, :model_not_found | :model_not_ready}
  def fetch_route(name, version) do
    case :ets.lookup(@table, {:route, name, version}) do
      [{_key, %Route{status: :ready} = route}] -> {:ok, route}
      [{_key, %Route{}}] -> {:error, :model_not_ready}
      [] -> {:error, :model_not_found}
    end
  end

  @doc "Fetches a route regardless of status. Used by lifecycle code, not by dispatch."
  @spec peek_route(atom(), String.t()) :: {:ok, Route.t()} | {:error, :model_not_found}
  def peek_route(name, version) do
    case :ets.lookup(@table, {:route, name, version}) do
      [{_key, %Route{} = route}] -> {:ok, route}
      [] -> {:error, :model_not_found}
    end
  end

  @doc "Returns every registered model name, sorted and deduplicated."
  @spec list() :: [atom()]
  def list do
    @table
    |> :ets.match({{:default, :"$1"}, :_})
    |> List.flatten()
    |> Enum.sort()
  end

  @doc "Returns every registered version of `name`, newest registration last."
  @spec versions(atom()) :: {:ok, [String.t()]} | {:error, :model_not_found}
  def versions(name) do
    case :ets.match(@table, {{:route, name, :"$1"}, :_}) do
      [] -> {:error, :model_not_found}
      versions -> {:ok, versions |> List.flatten() |> Enum.sort(&version_lte?/2)}
    end
  end

  @doc "Returns the version unpinned traffic is routed to."
  @spec default_version(atom()) :: {:ok, String.t()} | {:error, :model_not_found}
  def default_version(name) do
    case :ets.lookup(@table, {:default, name}) do
      [{_key, version}] -> {:ok, version}
      [] -> {:error, :model_not_found}
    end
  end

  @doc "Returns the active canary as `{version, weight}`, or `nil`."
  @spec canary(atom()) :: {String.t(), 1..100} | nil
  def canary(name) do
    case :ets.lookup(@table, {:canary, name}) do
      [{_key, canary}] -> canary
      [] -> nil
    end
  end

  @doc "Returns the full status entry for a model version."
  @spec status(atom(), String.t()) :: {:ok, map()} | {:error, :model_not_found}
  def status(name, version) do
    case :ets.lookup(@table, {:model, name, version}) do
      [{_key, entry}] -> {:ok, entry}
      [] -> {:error, :model_not_found}
    end
  end

  @doc false
  @spec register(ModelSpec.t()) :: {:ok, Route.t()} | {:error, term()}
  def register(%ModelSpec{} = spec), do: GenServer.call(__MODULE__, {:register, spec})

  @doc false
  @spec update_status(atom(), String.t(), atom(), keyword()) :: :ok
  def update_status(name, version, status, extra \\ []) do
    GenServer.call(__MODULE__, {:update_status, name, version, status, extra})
  end

  @doc false
  @spec unregister(atom(), String.t()) :: :ok
  def unregister(name, version), do: GenServer.call(__MODULE__, {:unregister, name, version})

  @doc false
  @spec set_default(atom(), String.t()) :: :ok | {:error, :model_not_found}
  def set_default(name, version), do: GenServer.call(__MODULE__, {:set_default, name, version})

  @doc false
  @spec set_canary(atom(), String.t(), 1..100) :: :ok | {:error, term()}
  def set_canary(name, version, weight) do
    GenServer.call(__MODULE__, {:set_canary, name, version, weight})
  end

  @doc false
  @spec clear_canary(atom()) :: :ok
  def clear_canary(name), do: GenServer.call(__MODULE__, {:clear_canary, name})

  @doc false
  @spec spec(atom(), String.t()) :: {:ok, ModelSpec.t()} | {:error, :model_not_found}
  def spec(name, version) do
    case status(name, version) do
      {:ok, %{spec: spec}} -> {:ok, spec}
      error -> error
    end
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    # :protected — only this process writes. read_concurrency because the read/write ratio here
    # is thousands of predictions per deploy-time write.
    table =
      :ets.new(@table, [
        :set,
        :protected,
        :named_table,
        read_concurrency: true,
        decentralized_counters: false
      ])

    {:ok, %{table: table}}
  end

  @impl true
  def handle_call({:register, spec}, _from, state) do
    key = {:route, spec.name, spec.version}

    case :ets.lookup(@table, key) do
      [{_key, %Route{status: status}}] when status != :failed ->
        {:reply, {:error, {:already_loaded, spec.name, spec.version}}, state}

      _ ->
        route = Route.from_spec(spec, status: :loading)

        :ets.insert(@table, {key, route})
        :ets.insert(@table, {{:model, spec.name, spec.version}, initial_entry(spec)})

        # First version registered for a name becomes the default, so a single-version model
        # never needs an explicit promote.
        unless :ets.member(@table, {:default, spec.name}) do
          :ets.insert(@table, {{:default, spec.name}, spec.version})
        end

        {:reply, {:ok, route}, state}
    end
  end

  def handle_call({:update_status, name, version, status, extra}, _from, state) do
    case :ets.lookup(@table, {:route, name, version}) do
      [{key, %Route{} = route}] ->
        # The status entry is written *before* the route, and the order is load-bearing. Readers
        # poll these two rows independently: `MLServe.ready?/1` reads the route while
        # `MLServe.model_status/2` reads the entry. Writing the route first opens a window where
        # a model reports ready but its status still says :loading, with no metadata — which is
        # exactly what a readiness probe or status page would show at the worst moment.
        update_entry(name, version, Map.new([{:status, status} | extra]))
        :ets.insert(@table, {key, %{route | status: status}})
        {:reply, :ok, state}

      [] ->
        {:reply, :ok, state}
    end
  end

  def handle_call({:unregister, name, version}, _from, state) do
    :ets.delete(@table, {:route, name, version})
    :ets.delete(@table, {:model, name, version})
    :persistent_term.erase(Route.state_key(name, version))

    case canary(name) do
      {^version, _weight} -> :ets.delete(@table, {:canary, name})
      _ -> :ok
    end

    remaining = remaining_versions(name)

    case {:ets.lookup(@table, {:default, name}), remaining} do
      {[{_key, ^version}], []} ->
        :ets.delete(@table, {:default, name})

      {[{_key, ^version}], [next | _]} ->
        # The default was just removed. Fall back to the newest surviving version rather than
        # leaving the name pointing at nothing — an unpinned predict/3 should keep working.
        Logger.info(
          "[ml_serve] #{inspect(name)} default version #{version} unloaded, falling back to #{next}"
        )

        :ets.insert(@table, {{:default, name}, next})

      _ ->
        :ok
    end

    {:reply, :ok, state}
  end

  def handle_call({:set_default, name, version}, _from, state) do
    if :ets.member(@table, {:route, name, version}) do
      :ets.insert(@table, {{:default, name}, version})
      :ets.delete(@table, {:canary, name})
      {:reply, :ok, state}
    else
      {:reply, {:error, :model_not_found}, state}
    end
  end

  def handle_call({:set_canary, name, version, weight}, _from, state) do
    cond do
      not :ets.member(@table, {:route, name, version}) ->
        {:reply, {:error, :model_not_found}, state}

      :ets.lookup(@table, {:default, name}) == [{{:default, name}, version}] ->
        {:reply, {:error, :already_default}, state}

      true ->
        :ets.insert(@table, {{:canary, name}, {version, weight}})
        {:reply, :ok, state}
    end
  end

  def handle_call({:clear_canary, name}, _from, state) do
    :ets.delete(@table, {:canary, name})
    {:reply, :ok, state}
  end

  # Private Functions

  defp resolve_default(name) do
    case :ets.lookup(@table, {:default, name}) do
      [] -> {:error, :model_not_found}
      [{_key, default}] -> resolve_canary(name, default, canary(name))
    end
  end

  defp resolve_canary(name, default, nil), do: fetch_route(name, default)

  defp resolve_canary(name, default, {candidate, weight}) do
    # The roll happens in the calling process against its own seed: no shared counter, no
    # coordination point, and therefore nothing to contend on at the split.
    if :rand.uniform(100) <= weight do
      route_canary(name, default, candidate)
    else
      fetch_route(name, default)
    end
  end

  defp route_canary(name, default, candidate) do
    case fetch_route(name, candidate) do
      {:ok, route} -> {:ok, %{route | canary?: true}}
      # A candidate that is not ready must not black-hole its share of traffic.
      {:error, _reason} -> fetch_route(name, default)
    end
  end

  defp initial_entry(spec) do
    %{
      name: spec.name,
      version: spec.version,
      status: :loading,
      spec: spec,
      backend: spec.backend,
      loaded_at: nil,
      load_duration_ms: nil,
      metadata: %{},
      failure: nil
    }
  end

  defp update_entry(name, version, changes) do
    case :ets.lookup(@table, {:model, name, version}) do
      [{key, entry}] -> :ets.insert(@table, {key, Map.merge(entry, changes)})
      [] -> :ok
    end
  end

  defp remaining_versions(name) do
    case versions(name) do
      {:ok, versions} -> Enum.reverse(versions)
      {:error, _} -> []
    end
  end

  # Sorts semver-ish strings numerically where possible so "10.0.0" follows "9.0.0", falling back
  # to string order for anything that is not dot-separated integers.
  defp version_lte?(a, b), do: version_key(a) <= version_key(b)

  defp version_key(version) do
    parts = String.split(version, [".", "-"])

    Enum.map(parts, fn part ->
      case Integer.parse(part) do
        {int, ""} -> {0, int, ""}
        _ -> {1, 0, part}
      end
    end)
  end
end
