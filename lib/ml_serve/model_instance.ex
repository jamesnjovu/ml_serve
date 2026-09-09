defmodule MLServe.ModelInstance do
  @moduledoc """
  The supervision subtree for one loaded model version.

  ```text
  MLServe.ModelInstance                (:rest_for_one)
  ├── MLServe.ModelServer              lifecycle, status, drain
  ├── MLServe.WorkerSupervisor         only when concurrency: :exclusive
  │     └── MLServe.Worker × N
  └── MLServe.Batcher                  only when batching is configured
  ```

  `:rest_for_one` is the correct strategy and the ordering is load-bearing. `MLServe.ModelServer`
  owns the loaded backend state that workers are handed; if it dies, the state its workers hold
  is stale, so the workers and batcher must restart behind it. The reverse is not true — a
  crashed worker has no bearing on the model server, so restarting only what follows is exactly
  right.

  The worker supervisor and batcher are started *by* `MLServe.ModelServer` once loading succeeds,
  rather than declared here, because neither can exist before there is state to give them.
  """

  use Supervisor

  alias MLServe.ModelSpec

  @doc false
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts) do
    spec = Keyword.fetch!(opts, :spec)
    Supervisor.start_link(__MODULE__, opts, name: via(spec.name, spec.version))
  end

  @doc false
  @spec via(atom(), String.t()) :: Supervisor.name()
  def via(name, version) do
    {:via, Registry, {MLServe.Registry, {:instance, name, version}}}
  end

  @doc false
  @spec whereis(atom(), String.t()) :: pid() | nil
  def whereis(name, version) do
    case Registry.lookup(MLServe.Registry, {:instance, name, version}) do
      [{pid, _value}] -> pid
      [] -> nil
    end
  end

  @doc false
  @spec child_spec(keyword()) :: Supervisor.child_spec()
  def child_spec(opts) do
    spec = Keyword.fetch!(opts, :spec)

    %{
      id: ModelSpec.key(spec),
      start: {__MODULE__, :start_link, [opts]},
      type: :supervisor,
      restart: :transient
    }
  end

  @impl true
  def init(opts) do
    Supervisor.init([{MLServe.ModelServer, opts}], strategy: :rest_for_one)
  end
end
