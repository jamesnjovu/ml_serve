defmodule MLServe.ModelSpec do
  @moduledoc """
  The normalised, validated description of one loaded model version.

  `MLServe.Config.build/2` turns a keyword list — from `config :ml_serve, :models` or from
  `MLServe.load_model/2` — into this struct. Everything downstream (the registry, the supervisor,
  the workers, the dispatcher) reads the struct rather than re-interpreting raw options, so there
  is exactly one place where option handling lives.

  A spec is immutable for the lifetime of a loaded model version. Mutable runtime facts — status,
  worker pids, request counters — live in `MLServe.ModelRegistry`, not here.
  """

  @typedoc "An `{module, function, extra_args}` hook invoked in the calling process."
  @type hook :: {module(), atom(), list()} | (term() -> term()) | nil

  @type t :: %__MODULE__{
          name: atom(),
          version: String.t(),
          backend: module(),
          config: keyword(),
          path: String.t() | nil,
          checksum: {:sha256 | :sha512, String.t()} | nil,
          workers: pos_integer(),
          concurrency: :shared | :exclusive,
          load: :once | :per_worker,
          timeout: timeout(),
          drain_timeout: timeout(),
          max_concurrency: pos_integer() | :infinity,
          max_batch_size: pos_integer(),
          batching: %{max_size: pos_integer(), timeout: pos_integer()} | nil,
          cache: %{enabled: boolean(), ttl: pos_integer()} | nil,
          preprocess: hook(),
          postprocess: hook(),
          restart_on_error: boolean(),
          selection: :round_robin | :least_loaded | :random
        }

  defstruct [
    :name,
    :version,
    :backend,
    :path,
    :checksum,
    :batching,
    :cache,
    :preprocess,
    :postprocess,
    config: [],
    workers: 1,
    concurrency: :exclusive,
    load: :once,
    timeout: 5_000,
    drain_timeout: 5_000,
    max_concurrency: :infinity,
    max_batch_size: 1_000,
    restart_on_error: false,
    selection: :round_robin
  ]

  @doc """
  Returns the registry key identifying this model version.

  ## Examples

      iex> spec = %MLServe.ModelSpec{name: :fraud, version: "1.0.0"}
      iex> MLServe.ModelSpec.key(spec)
      {:fraud, "1.0.0"}
  """
  @spec key(t()) :: {atom(), String.t()}
  def key(%__MODULE__{name: name, version: version}), do: {name, version}

  @doc """
  Returns true when this model runs inference in the calling process rather than in a worker pool.

  ## Examples

      iex> MLServe.ModelSpec.shared?(%MLServe.ModelSpec{concurrency: :shared})
      true

      iex> MLServe.ModelSpec.shared?(%MLServe.ModelSpec{concurrency: :exclusive})
      false
  """
  @spec shared?(t()) :: boolean()
  def shared?(%__MODULE__{concurrency: :shared}), do: true
  def shared?(%__MODULE__{}), do: false

  @doc """
  Returns the number of worker processes this model will start.

  Shared-concurrency models start none: inference runs in the caller.

  ## Examples

      iex> MLServe.ModelSpec.worker_count(%MLServe.ModelSpec{concurrency: :shared, workers: 8})
      0

      iex> MLServe.ModelSpec.worker_count(%MLServe.ModelSpec{concurrency: :exclusive, workers: 8})
      8
  """
  @spec worker_count(t()) :: non_neg_integer()
  def worker_count(%__MODULE__{concurrency: :shared}), do: 0
  def worker_count(%__MODULE__{workers: workers}), do: workers
end
