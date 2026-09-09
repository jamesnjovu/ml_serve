defmodule MLServe.Route do
  @moduledoc """
  The slim record read from ETS on every prediction.

  Deliberately separate from `MLServe.ModelSpec`. A spec carries the backend's `:config` — which
  may hold closures, large keyword lists, or preprocessing tables — and reading it out of ETS
  copies the whole thing into the calling process. A route carries only what dispatch needs, so
  the per-prediction copy stays at a few dozen words.

  Everything here is either an atom, a small integer, or a reference. Nothing that grows with
  model size ever enters this struct.
  """

  @type t :: %__MODULE__{
          name: atom(),
          version: String.t(),
          backend: module(),
          status: :loading | :ready | :draining | :failed | :unloading,
          concurrency: :shared | :exclusive,
          workers: non_neg_integer(),
          selection: :round_robin | :least_loaded | :random,
          timeout: timeout(),
          max_concurrency: pos_integer() | :infinity,
          max_batch_size: pos_integer(),
          batching: boolean(),
          cache: %{enabled: boolean(), ttl: pos_integer()} | nil,
          preprocess: MLServe.ModelSpec.hook(),
          postprocess: MLServe.ModelSpec.hook(),
          restart_on_error: boolean(),
          counters: :counters.counters_ref() | nil,
          atomics: :atomics.atomics_ref() | nil,
          canary?: boolean()
        }

  defstruct [
    :name,
    :version,
    :backend,
    :cache,
    :preprocess,
    :postprocess,
    :counters,
    :atomics,
    status: :loading,
    concurrency: :exclusive,
    workers: 1,
    selection: :round_robin,
    timeout: 5_000,
    max_concurrency: :infinity,
    max_batch_size: 1_000,
    batching: false,
    restart_on_error: false,
    canary?: false
  ]

  # Counter slots. in_flight is read exactly (for drain), requests and errors are write-heavy
  # and read rarely, which is precisely what :counters with write_concurrency is built for.
  @in_flight 1
  @requests 2
  @errors 3

  # Atomics slot. Round-robin needs an atomic fetch-and-add returning the new value, which
  # :counters cannot do and :atomics.add_get/3 can.
  @rr_index 1

  @doc false
  @spec new_counters() :: :counters.counters_ref()
  def new_counters, do: :counters.new(3, [:write_concurrency])

  @doc false
  @spec new_atomics() :: :atomics.atomics_ref()
  def new_atomics, do: :atomics.new(1, signed: false)

  @doc "Builds a route from a validated spec."
  @spec from_spec(MLServe.ModelSpec.t(), keyword()) :: t()
  def from_spec(%MLServe.ModelSpec{} = spec, opts \\ []) do
    %__MODULE__{
      name: spec.name,
      version: spec.version,
      backend: spec.backend,
      status: Keyword.get(opts, :status, :loading),
      concurrency: spec.concurrency,
      workers: MLServe.ModelSpec.worker_count(spec),
      selection: spec.selection,
      timeout: spec.timeout,
      max_concurrency: spec.max_concurrency,
      max_batch_size: spec.max_batch_size,
      batching: spec.batching != nil,
      cache: spec.cache,
      preprocess: spec.preprocess,
      postprocess: spec.postprocess,
      restart_on_error: spec.restart_on_error,
      counters: Keyword.get(opts, :counters) || new_counters(),
      atomics: Keyword.get(opts, :atomics) || new_atomics()
    }
  end

  @doc "Atomically increments and returns the next round-robin worker index, 0-based."
  @spec next_index(t()) :: non_neg_integer()
  def next_index(%__MODULE__{atomics: atomics, workers: workers}) when workers > 0 do
    rem(:atomics.add_get(atomics, @rr_index, 1), workers)
  end

  @doc "Increments the in-flight request count."
  @spec enter(t()) :: :ok
  def enter(%__MODULE__{counters: counters}) do
    :counters.add(counters, @in_flight, 1)
    :counters.add(counters, @requests, 1)
  end

  @doc "Decrements the in-flight request count, recording an error when `ok?` is false."
  @spec leave(t(), boolean()) :: :ok
  def leave(%__MODULE__{counters: counters}, ok?) do
    :counters.sub(counters, @in_flight, 1)
    unless ok?, do: :counters.add(counters, @errors, 1)
    :ok
  end

  @doc "Current number of in-flight requests."
  @spec in_flight(t() | :counters.counters_ref()) :: non_neg_integer()
  def in_flight(%__MODULE__{counters: counters}), do: in_flight(counters)
  def in_flight(counters), do: :counters.get(counters, @in_flight)

  @doc "Total requests accepted since the model was loaded."
  @spec requests(t() | :counters.counters_ref()) :: non_neg_integer()
  def requests(%__MODULE__{counters: counters}), do: requests(counters)
  def requests(counters), do: :counters.get(counters, @requests)

  @doc "Total failed requests since the model was loaded."
  @spec errors(t() | :counters.counters_ref()) :: non_neg_integer()
  def errors(%__MODULE__{counters: counters}), do: errors(counters)
  def errors(counters), do: :counters.get(counters, @errors)

  @doc """
  Reserves an admission slot, returning `:ok` or `{:error, :overloaded}`.

  Admission control is a compare-then-increment on the in-flight counter with no lock and no
  process. Under a race two callers can both observe `max - 1` and both proceed, so the limit is
  approximate at the boundary. That is the right trade: an exact limiter would need serialisation
  through a process, which is the very bottleneck the limit exists to prevent.
  """
  @spec admit(t()) :: :ok | {:error, :overloaded}
  def admit(%__MODULE__{max_concurrency: :infinity} = route) do
    enter(route)
    :ok
  end

  def admit(%__MODULE__{max_concurrency: max} = route) do
    if in_flight(route) >= max do
      :counters.add(route.counters, @errors, 1)
      {:error, :overloaded}
    else
      enter(route)
      :ok
    end
  end

  @doc "The `:persistent_term` key under which a shared backend's state is stored."
  @spec state_key(atom(), String.t()) :: {module(), :state, atom(), String.t()}
  def state_key(name, version), do: {MLServe, :state, name, version}
end
