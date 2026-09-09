defmodule MLServe.Config do
  @moduledoc """
  Reads and validates MLServe configuration.

  Two jobs: expose the application-wide settings as typed accessors, and turn a per-model keyword
  list into a validated `MLServe.ModelSpec`. The same `build/2` path serves both static
  configuration and runtime `MLServe.load_model/2`, so a model registered at runtime is validated
  exactly as strictly as one declared in `config.exs`.

  Validation is hand-rolled rather than delegated to a schema library. MLServe has one runtime
  dependency on purpose, and a few hundred lines of explicit validation buys clearer error
  messages than a generic schema violation.

  ## Application configuration

      config :ml_serve,
        model_root: "priv/models",
        start_mode: :async,
        default_timeout: 5_000,
        max_batch_size: 1_000,
        max_model_bytes: 2_147_483_648,
        cache: [enabled: true, max_size: 10_000, ttl: :timer.minutes(5)],
        models: [
          fraud_detection: [backend: MyApp.Backends.ONNX, path: "fraud.onnx", workers: 4]
        ]

  ## Model options

  | Option | Default | Meaning |
  | ------ | ------- | ------- |
  | `:backend` | *required* | Module implementing `MLServe.Model` |
  | `:version` | `"1.0.0"` | Version string; models are keyed by `{name, version}` |
  | `:path` | `nil` | Model artifact, validated by `MLServe.Security` |
  | `:checksum` | `nil` | `{:sha256, hex}` integrity check |
  | `:workers` | `System.schedulers_online()` | Pool size; ignored for `:shared` backends |
  | `:timeout` | `:default_timeout` | Per-request inference timeout |
  | `:drain_timeout` | `5_000` | How long `unload_model/2` waits for in-flight requests |
  | `:max_concurrency` | `:infinity` | Admission limit; excess returns `{:error, :overloaded}` |
  | `:max_batch_size` | `:max_batch_size` | Largest accepted `batch_predict/3` list |
  | `:batching` | `nil` | `[max_size: 16, timeout: 10]` enables dynamic batching |
  | `:cache` | `nil` | `[enabled: true, ttl: 60_000]` |
  | `:preprocess` / `:postprocess` | `nil` | `{Mod, :fun, args}` hooks run in the caller |
  | `:restart_on_error` | `false` | Crash the worker on a backend exception instead of returning it |
  | `:selection` | `:round_robin` | `:round_robin`, `:least_loaded` or `:random` |
  | `:config` | `[]` | Opaque keyword list passed to the backend's `load/1` |
  """

  alias MLServe.Error
  alias MLServe.ModelSpec
  alias MLServe.Security

  @default_timeout 5_000
  @default_drain_timeout 5_000
  @default_max_batch_size 1_000
  @default_cache_ttl :timer.minutes(5)
  @default_cache_max_size 10_000
  @default_max_model_bytes 2 * 1024 * 1024 * 1024
  @default_version "1.0.0"

  @model_keys ~w(backend version path checksum workers concurrency load timeout drain_timeout
                 max_concurrency max_batch_size batching cache preprocess postprocess
                 restart_on_error selection config)a

  # Application configuration

  @doc "Root directory model paths must resolve inside. Defaults to `priv/models` of the app."
  @spec model_root() :: String.t()
  def model_root do
    get(:model_root) || Path.join(File.cwd!(), "priv/models")
  end

  @doc "Whether models declared in configuration load in the background (`:async`) or block boot."
  @spec start_mode() :: :async | :sync
  def start_mode, do: get(:start_mode) || :async

  @doc "Default per-request inference timeout in milliseconds."
  @spec default_timeout() :: timeout()
  def default_timeout, do: get(:default_timeout) || @default_timeout

  @doc "Default upper bound on `MLServe.batch_predict/3` list length."
  @spec max_batch_size() :: pos_integer()
  def max_batch_size, do: get(:max_batch_size) || @default_max_batch_size

  @doc "Largest model artifact `MLServe.Security` will accept, in bytes."
  @spec max_model_bytes() :: pos_integer() | :infinity
  def max_model_bytes, do: get(:max_model_bytes) || @default_max_model_bytes

  @doc "Inference cache settings."
  @spec cache() :: %{
          enabled: boolean(),
          max_size: pos_integer(),
          ttl: pos_integer(),
          sweep_interval: pos_integer()
        }
  def cache do
    opts = get(:cache) || []

    %{
      enabled: Keyword.get(opts, :enabled, true),
      max_size: Keyword.get(opts, :max_size, @default_cache_max_size),
      ttl: Keyword.get(opts, :ttl, @default_cache_ttl),
      sweep_interval: Keyword.get(opts, :sweep_interval, :timer.minutes(1))
    }
  end

  @doc "Models declared in application configuration, as `{name, opts}` pairs."
  @spec configured_models() :: keyword()
  def configured_models, do: get(:models) || []

  # Model specs

  @doc """
  Builds a validated `MLServe.ModelSpec` from a model name and options.

  ## Parameters

    - `name`: the model's atom name
    - `opts`: the model options documented in the moduledoc

  ## Examples

      iex> {:ok, spec} = MLServe.Config.build(:fraud, backend: MLServe.Backend.Static, config: [result: :ok])
      iex> {spec.name, spec.version, spec.backend}
      {:fraud, "1.0.0", MLServe.Backend.Static}
  """
  @spec build(atom(), keyword()) :: {:ok, ModelSpec.t()} | {:error, Error.t()}
  def build(name, opts) when is_atom(name) and is_list(opts) do
    with :ok <- validate_name(name),
         :ok <- validate_known_keys(name, opts),
         {:ok, backend} <- fetch_backend(name, opts),
         {:ok, version} <- fetch_version(name, opts),
         {:ok, capabilities} <- capabilities(name, backend, opts),
         {:ok, path} <- fetch_path(name, opts),
         {:ok, workers} <- fetch_workers(name, opts),
         {:ok, batching} <- fetch_batching(name, opts),
         {:ok, cache} <- fetch_cache(name, opts),
         {:ok, preprocess} <- fetch_hook(name, opts, :preprocess),
         {:ok, postprocess} <- fetch_hook(name, opts, :postprocess),
         {:ok, selection} <- fetch_selection(name, opts),
         {:ok, max_concurrency} <- fetch_max_concurrency(name, opts) do
      {:ok,
       %ModelSpec{
         name: name,
         version: version,
         backend: backend,
         path: path,
         checksum: Keyword.get(opts, :checksum),
         config:
           Keyword.get(opts, :config, [])
           |> Keyword.put(:version, version)
           |> maybe_put_path(path),
         workers: workers,
         concurrency: capabilities.concurrency,
         load: capabilities.load,
         timeout: Keyword.get(opts, :timeout, default_timeout()),
         drain_timeout: Keyword.get(opts, :drain_timeout, @default_drain_timeout),
         max_concurrency: max_concurrency,
         max_batch_size: Keyword.get(opts, :max_batch_size, max_batch_size()),
         batching: batching,
         cache: cache,
         preprocess: preprocess,
         postprocess: postprocess,
         restart_on_error: Keyword.get(opts, :restart_on_error, false),
         selection: selection
       }}
    end
  end

  @doc """
  Same as `build/2` but raises `MLServe.Error` on invalid configuration.
  """
  @spec build!(atom(), keyword()) :: ModelSpec.t()
  def build!(name, opts) do
    case build(name, opts) do
      {:ok, spec} -> spec
      {:error, error} -> raise error
    end
  end

  # Private Functions

  defp get(key), do: Application.get_env(:ml_serve, key)

  defp validate_name(name) when is_atom(name) and not is_nil(name), do: :ok

  defp validate_known_keys(name, opts) do
    case Enum.reject(Keyword.keys(opts), &(&1 in @model_keys)) do
      [] ->
        :ok

      unknown ->
        invalid(
          name,
          "unknown option#{if length(unknown) > 1, do: "s"} #{Enum.map_join(unknown, ", ", &inspect/1)}. " <>
            "Valid options: #{Enum.map_join(@model_keys, ", ", &inspect/1)}"
        )
    end
  end

  defp fetch_backend(name, opts) do
    case Keyword.get(opts, :backend) do
      nil ->
        invalid(name, "the :backend option is required")

      backend ->
        case Security.validate_backend(backend) do
          :ok ->
            {:ok, backend}

          {:error, {:invalid_backend, :not_loaded}} ->
            invalid(name, ":backend #{inspect(backend)} is not a loaded module")

          {:error, {:invalid_backend, :not_a_model}} ->
            invalid(
              name,
              ":backend #{inspect(backend)} does not implement MLServe.Model " <>
                "(load/1 and predict/2 are required)"
            )
        end
    end
  end

  defp fetch_version(name, opts) do
    case Keyword.get(opts, :version, @default_version) do
      version when is_binary(version) and byte_size(version) > 0 -> {:ok, version}
      other -> invalid(name, ":version must be a non-empty string, got #{inspect(other)}")
    end
  end

  defp capabilities(name, backend, opts) do
    declared =
      if function_exported?(backend, :capabilities, 0) do
        backend.capabilities()
      else
        %{}
      end

    concurrency = Keyword.get(opts, :concurrency) || Map.get(declared, :concurrency, :exclusive)
    load = Keyword.get(opts, :load) || Map.get(declared, :load, :once)

    cond do
      concurrency not in [:shared, :exclusive] ->
        invalid(name, ":concurrency must be :shared or :exclusive, got #{inspect(concurrency)}")

      load not in [:once, :per_worker] ->
        invalid(name, ":load must be :once or :per_worker, got #{inspect(load)}")

      concurrency == :shared and load == :per_worker ->
        invalid(
          name,
          "load: :per_worker is meaningless with concurrency: :shared, which starts no workers"
        )

      true ->
        {:ok, %{concurrency: concurrency, load: load}}
    end
  end

  defp fetch_path(name, opts) do
    case Keyword.get(opts, :path) do
      nil -> {:ok, nil}
      path when is_binary(path) -> {:ok, path}
      other -> invalid(name, ":path must be a string, got #{inspect(other)}")
    end
  end

  defp fetch_workers(name, opts) do
    case Keyword.get(opts, :workers, System.schedulers_online()) do
      workers when is_integer(workers) and workers > 0 -> {:ok, workers}
      other -> invalid(name, ":workers must be a positive integer, got #{inspect(other)}")
    end
  end

  defp fetch_batching(name, opts) do
    case Keyword.get(opts, :batching) do
      nil ->
        {:ok, nil}

      false ->
        {:ok, nil}

      batching when is_list(batching) ->
        {:ok,
         %{
           max_size: Keyword.get(batching, :max_size, 16),
           timeout: Keyword.get(batching, :timeout, 10)
         }}

      other ->
        invalid(name, ":batching must be a keyword list or false, got #{inspect(other)}")
    end
  end

  defp fetch_cache(name, opts) do
    case Keyword.get(opts, :cache) do
      nil ->
        {:ok, nil}

      false ->
        {:ok, %{enabled: false, ttl: cache().ttl}}

      true ->
        {:ok, %{enabled: true, ttl: cache().ttl}}

      opts when is_list(opts) ->
        {:ok,
         %{
           enabled: Keyword.get(opts, :enabled, true),
           ttl: Keyword.get(opts, :ttl, cache().ttl)
         }}

      other ->
        invalid(name, ":cache must be a boolean or keyword list, got #{inspect(other)}")
    end
  end

  defp fetch_hook(name, opts, key) do
    case Keyword.get(opts, key) do
      nil ->
        {:ok, nil}

      {mod, fun, args} when is_atom(mod) and is_atom(fun) and is_list(args) ->
        {:ok, {mod, fun, args}}

      fun when is_function(fun, 1) ->
        {:ok, fun}

      other ->
        invalid(
          name,
          "#{inspect(key)} must be a {module, function, args} tuple or a 1-arity function, " <>
            "got #{inspect(other)}"
        )
    end
  end

  defp fetch_selection(name, opts) do
    case Keyword.get(opts, :selection, :round_robin) do
      selection when selection in [:round_robin, :least_loaded, :random] ->
        {:ok, selection}

      other ->
        invalid(
          name,
          ":selection must be :round_robin, :least_loaded or :random, got #{inspect(other)}"
        )
    end
  end

  defp fetch_max_concurrency(name, opts) do
    case Keyword.get(opts, :max_concurrency, :infinity) do
      :infinity ->
        {:ok, :infinity}

      max when is_integer(max) and max > 0 ->
        {:ok, max}

      other ->
        invalid(
          name,
          ":max_concurrency must be a positive integer or :infinity, got #{inspect(other)}"
        )
    end
  end

  defp maybe_put_path(config, nil), do: config
  defp maybe_put_path(config, path), do: Keyword.put(config, :path, path)

  defp invalid(name, message) do
    {:error,
     Error.new(:config, "invalid configuration for model #{inspect(name)}: #{message}",
       details: %{model: name}
     )}
  end
end
