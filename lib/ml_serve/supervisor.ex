defmodule MLServe.Supervisor do
  @moduledoc """
  MLServe's top-level supervision tree.

  ```text
  MLServe.Supervisor                     (:rest_for_one)
  ├── MLServe.Registry                   process registry, partitioned by scheduler
  ├── MLServe.ModelRegistry              owns the catalog ETS table
  ├── MLServe.Cache                      owns the cache ETS table + sweeper
  ├── MLServe.TaskSupervisor             batch fan-out
  └── MLServe.ModelSupervisor            one subtree per loaded {name, version}
  ```

  The strategy is `:rest_for_one`, and the child order is the reason. `MLServe.ModelRegistry`
  owns the catalog ETS table; ETS tables die with their owner. If that process restarts, every
  route in the system has just evaporated, and any model subtree still running would be serving
  traffic that the registry no longer knows about. Restarting everything after it is the only
  consistent answer. `:one_for_one` would leave orphaned models behind.

  Conversely a crashed model subtree has no bearing on the registry, so nothing above it
  restarts — which is what per-model isolation means in practice.
  """

  use Supervisor

  @doc false
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(opts) do
    children = [
      # Partitioned so worker lookups on the hot path do not contend on a single ETS table.
      {Registry, keys: :unique, name: MLServe.Registry, partitions: System.schedulers_online()},
      MLServe.ModelRegistry,
      MLServe.Cache,
      {Task.Supervisor, name: MLServe.TaskSupervisor},
      MLServe.ModelSupervisor,
      {MLServe.Bootstrap, opts}
    ]

    Supervisor.init(children, strategy: :rest_for_one)
  end
end
