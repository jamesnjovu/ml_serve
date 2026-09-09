defmodule MLServe.Worker do
  @moduledoc """
  A single inference worker holding backend state.

  Started only for models whose backend declares `concurrency: :exclusive`. Shared backends run
  in the calling process and never reach this module.

  A worker does exactly one thing: call the backend. Cache lookups, input validation and the
  `preprocess`/`postprocess` hooks all run in the *caller* before dispatch, because a worker slot
  is the scarce resource — occupying one with an Ecto query while a GPU sits idle is the mistake
  this design exists to prevent.

  ## Deadlines

  Requests carry an absolute monotonic deadline, and the worker checks it *before* invoking the
  backend. Work whose caller has already timed out is dropped rather than run.

  That matters under overload. A `GenServer.call` timeout only abandons the caller's side; the
  worker still grinds through the whole queue, every item arriving later than the last, and the
  system never recovers. Checking the deadline at the front of the queue turns a death spiral
  into load shedding.
  """

  use GenServer

  require Logger

  alias MLServe.Backend
  alias MLServe.ModelSpec

  @type request :: {:predict, term(), integer()} | {:batch_predict, [term()], integer()}

  defstruct [:spec, :state, :index]

  # Client API

  @doc false
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    spec = Keyword.fetch!(opts, :spec)
    index = Keyword.fetch!(opts, :index)
    GenServer.start_link(__MODULE__, opts, name: via(spec.name, spec.version, index))
  end

  @doc false
  @spec via(atom(), String.t(), non_neg_integer()) :: GenServer.name()
  def via(name, version, index) do
    {:via, Registry, {MLServe.Registry, {:worker, name, version, index}}}
  end

  @doc false
  @spec whereis(atom(), String.t(), non_neg_integer()) :: pid() | nil
  def whereis(name, version, index) do
    case Registry.lookup(MLServe.Registry, {:worker, name, version, index}) do
      [{pid, _value}] -> pid
      [] -> nil
    end
  end

  @doc """
  Runs a single prediction on `worker`, honouring `deadline`.

  Returns `{:ok, result, measurements}` so the caller can fold `queue_duration` and
  `inference_duration` into its telemetry span — the worker is the only place that can measure
  how long the request actually waited.
  """
  @spec predict(pid() | GenServer.name(), term(), integer(), timeout()) ::
          {:ok, term(), map()} | {:error, term()}
  def predict(worker, input, deadline, timeout) do
    call(worker, {:predict, input, System.monotonic_time(), deadline}, timeout)
  end

  @doc """
  Runs a batch prediction on `worker`, honouring `deadline`.
  """
  @spec batch_predict(pid() | GenServer.name(), [term()], integer(), timeout()) ::
          {:ok, [term()], map()} | {:error, term()}
  def batch_predict(worker, inputs, deadline, timeout) do
    call(worker, {:batch_predict, inputs, System.monotonic_time(), deadline}, timeout)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    spec = Keyword.fetch!(opts, :spec)
    index = Keyword.fetch!(opts, :index)

    # See MLServe.ModelServer: terminate/2 only runs when exits are trapped, and per-worker
    # backends must release their own state on shutdown.
    Process.flag(:trap_exit, true)

    case worker_state(spec, opts) do
      {:ok, state} ->
        {:ok, %__MODULE__{spec: spec, state: state, index: index}}

      {:error, reason} ->
        # A worker that cannot load its own state must not restart-loop against a broken model
        # file. :normal stops it quietly; ModelServer is what reports and retries the load.
        Logger.error(
          "[ml_serve] #{inspect(spec.name)} worker #{index} failed to load: #{inspect(reason)}"
        )

        :ignore
    end
  end

  @impl true
  def handle_call({kind, payload, enqueued_at, deadline}, _from, %__MODULE__{} = worker) do
    started_at = System.monotonic_time()

    if expired?(deadline, started_at) do
      {:reply, {:error, :timeout}, worker}
    else
      {result, finished_at} = invoke(kind, worker, payload)

      measurements = %{
        queue_duration: started_at - enqueued_at,
        inference_duration: finished_at - started_at
      }

      reply(result, measurements, worker)
    end
  end

  @impl true
  def terminate(_reason, %__MODULE__{spec: %ModelSpec{load: :per_worker} = spec, state: state}) do
    Backend.unload(spec, state)
  end

  def terminate(_reason, _worker), do: :ok

  # Private Functions

  # Per-worker loading gives each worker independent state (ports, sessions). :once means the
  # state was loaded by ModelServer and handed over — for NIF-resource models the term is a cheap
  # handle, so all workers share the underlying memory rather than each loading a copy.
  defp worker_state(%ModelSpec{load: :per_worker} = spec, _opts), do: Backend.load(spec)
  defp worker_state(%ModelSpec{}, opts), do: {:ok, Keyword.fetch!(opts, :state)}

  defp invoke(:predict, %__MODULE__{spec: spec, state: state}, input) do
    {Backend.predict(spec, state, input), System.monotonic_time()}
  end

  defp invoke(:batch_predict, %__MODULE__{spec: spec, state: state}, inputs) do
    {Backend.batch_predict(spec, state, inputs), System.monotonic_time()}
  end

  defp reply({:ok, result}, measurements, worker) do
    {:reply, {:ok, result, measurements}, worker}
  end

  defp reply({:error, {:backend_error, _} = reason}, _measurements, worker) do
    if worker.spec.restart_on_error do
      # Opt-in for backends whose state may be corrupt after a failure — a port that died, a
      # session left mid-transaction. The supervisor restarts the worker with fresh state and the
      # caller still receives the error rather than a bare :noproc.
      {:stop, {:shutdown, reason}, {:error, reason}, worker}
    else
      {:reply, {:error, reason}, worker}
    end
  end

  defp reply({:error, _} = error, _measurements, worker), do: {:reply, error, worker}

  defp expired?(deadline, now), do: now >= deadline

  # Every exit from the call is caught, not an enumerated few. A worker killed *while serving*
  # exits the caller with `:killed`, and an unmatched clause there would take a Phoenix request
  # process down with the worker — destroying exactly the isolation this library exists to give.
  #
  # Only a timeout is about this request. Every other exit reason means the worker is gone, so
  # the answer is :worker_gone and the dispatcher retries on another worker.
  defp call(worker, message, timeout) do
    GenServer.call(worker, message, timeout)
  catch
    :exit, {:timeout, _} -> {:error, :timeout}
    :exit, _reason -> {:error, :worker_gone}
  end
end
