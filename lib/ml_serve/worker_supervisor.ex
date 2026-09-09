defmodule MLServe.WorkerSupervisor do
  @moduledoc """
  Supervises one model version's worker pool.

  One of these per loaded `{name, version}` with an `:exclusive` backend. Shared-concurrency
  models start no workers, so no supervisor is started for them either.

  Scoping the pool to a single model version is what keeps failures isolated: a backend that
  crash-loops exhausts *its own* restart intensity and takes down *its own* model instance, while
  every other model keeps serving. A single global pool would let one bad model's restarts either
  starve or kill unrelated ones.
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
    {:via, Registry, {MLServe.Registry, {:workers, name, version}}}
  end

  @impl true
  def init(opts) do
    spec = Keyword.fetch!(opts, :spec)
    state = Keyword.get(opts, :state)

    children =
      for index <- 0..(ModelSpec.worker_count(spec) - 1)//1 do
        Supervisor.child_spec({MLServe.Worker, spec: spec, index: index, state: state},
          id: {:worker, index}
        )
      end

    # max_restarts is generous relative to the pool size: a transient failure in one worker
    # should not tear down a healthy pool, while a genuinely broken backend still gives up and
    # lets ModelServer mark the model :failed rather than restarting forever.
    Supervisor.init(children,
      strategy: :one_for_one,
      max_restarts: 3 * ModelSpec.worker_count(spec),
      max_seconds: 5
    )
  end
end
