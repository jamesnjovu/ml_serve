defmodule MLServe.ModelSupervisor do
  @moduledoc """
  `DynamicSupervisor` holding one `MLServe.ModelInstance` per loaded `{name, version}`.

  Models arrive and leave at runtime — from configuration at boot, from `MLServe.load_model/2`
  during a deploy, from a version promotion — which is exactly what a dynamic supervisor is for.
  Because each child is a whole subtree, two versions of the same model coexist as independent
  siblings, which is what makes canary rollout and drain-on-unload possible.
  """

  use DynamicSupervisor

  alias MLServe.ModelInstance
  alias MLServe.ModelSpec

  @doc false
  @spec start_link(keyword()) :: Supervisor.on_start()
  def start_link(opts) do
    DynamicSupervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Starts the supervision subtree for a model version."
  @spec start_model(ModelSpec.t()) :: DynamicSupervisor.on_start_child()
  def start_model(%ModelSpec{} = spec) do
    DynamicSupervisor.start_child(__MODULE__, {ModelInstance, spec: spec})
  end

  @doc "Stops the supervision subtree for a model version."
  @spec stop_model(atom(), String.t()) :: :ok | {:error, :not_found}
  def stop_model(name, version) do
    case ModelInstance.whereis(name, version) do
      nil -> {:error, :not_found}
      pid -> DynamicSupervisor.terminate_child(__MODULE__, pid)
    end
  end

  @impl true
  def init(_opts) do
    DynamicSupervisor.init(strategy: :one_for_one)
  end
end
