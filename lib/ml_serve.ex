defmodule MLServe do
  @moduledoc """
  Production machine-learning inference for the BEAM.

  MLServe is the runtime layer around a model, not a modelling library. It does not train, it
  does not own a tensor type, and it does not wrap an LLM HTTP API. It supplies the parts every
  team ends up rebuilding by hand when they put a model into a Phoenix application: supervision,
  worker pools, batching, caching, telemetry, versioning and safe rollout.

      MLServe.predict(:fraud_detection, %{
        amount: 1500.50,
        transaction_count_24h: 8,
        failed_transactions_24h: 2
      })
      #=> {:ok, %{prediction: :fraud, probability: 0.94}}

  ## Why the BEAM is the right place for this

  Inference serving is a concurrency problem wearing a machine-learning hat. Requests arrive in
  parallel, models are stateful and expensive to load, some backends are not thread-safe, a model
  that dies must not take the node with it, and swapping a model version must not require a
  deploy. That list is OTP's home ground.

  What that yields concretely:

    * **Per-model isolation.** Every loaded model version is its own supervision subtree. A
      backend that crash-loops exhausts its own restart budget and marks *itself* failed while
      every other model keeps serving.
    * **Zero-overhead dispatch.** Model lookup is one lock-free ETS read in the calling process.
      Nothing in MLServe serialises your traffic — see `MLServe.Dispatcher`.
    * **Shared-state backends cost nothing.** A backend that is safe to call concurrently, such
      as an `Nx.Serving`, runs *in the caller* with state read from `:persistent_term`. No worker
      processes, no message copies of your tensors.
    * **Zero-downtime model upgrades.** Load a new version beside the running one, send it a
      percentage of traffic, compare per-version telemetry, promote, drain the old one.

  ## Installation

      def deps do
        [{:ml_serve, "~> 0.1.0"}]
      end

  MLServe's only runtime dependency is `:telemetry`.

  ## Quick start

  Any module with `load/1` and `predict/2` is a model:

      defmodule MyApp.FraudModel do
        @behaviour MLServe.Model

        @impl true
        def load(config), do: {:ok, Keyword.fetch!(config, :threshold)}

        @impl true
        def predict(threshold, %{amount: amount}) do
          probability = min(amount / 2000, 1.0)
          {:ok, %{prediction: (if probability > threshold, do: :fraud, else: :legitimate),
                  probability: probability}}
        end
      end

  Register it in configuration:

      config :ml_serve,
        models: [
          fraud_detection: [
            backend: MyApp.FraudModel,
            version: "1.0.0",
            workers: 4,
            config: [threshold: 0.7]
          ]
        ]

  Or at runtime, which is the same code path:

      MLServe.load_model(:fraud_detection, backend: MyApp.FraudModel, config: [threshold: 0.7])

  Then predict from anywhere — a controller, an Oban job, a `Task`:

      case MLServe.predict(:fraud_detection, features) do
        {:ok, result} -> result
        {:error, reason} -> Logger.warning("inference failed: \#{inspect(reason)}")
      end

  ## Errors

  Every function here returns `{:ok, result}` or `{:error, reason}`. Bang variants raise
  `MLServe.Error` instead.

  | Reason | Meaning |
  | ------ | ------- |
  | `:model_not_found` | No model registered under that name (or version) |
  | `:model_not_ready` | Registered, but loading, draining or failed |
  | `:timeout` | The deadline passed before inference completed |
  | `:overloaded` | The model is at its `:max_concurrency` limit |
  | `{:invalid_input, reason}` | Rejected by a `:preprocess` hook |
  | `{:batch_too_large, max}` | Batch exceeded `:max_batch_size` |
  | `{:backend_error, %MLServe.BackendError{}}` | The backend raised; exception and stacktrace preserved |
  | `{:load_failed, reason}` | The model could not be loaded |

  `MLServe.Error.retryable?/1` distinguishes transient failures from permanent ones.

  ## Guides

    * [Getting Started](getting-started.md) · [Architecture](architecture.md)
    * [Creating a Model Backend](creating-a-backend.md) — complete Nx, Bumblebee, ONNX and Python
      implementations
    * [Running Inference](running-inference.md) · [Batch Inference](batch-inference.md) ·
      [Concurrency](concurrency.md)
    * [Telemetry](telemetry.md) · [Model Versioning](model-versioning.md)
    * [Phoenix](phoenix-integration.md) · [Oban](oban-integration.md) ·
      [Production Deployment](production-deployment.md)
  """

  alias MLServe.Backend
  alias MLServe.Config
  alias MLServe.Dispatcher
  alias MLServe.Error
  alias MLServe.ModelRegistry
  alias MLServe.ModelServer
  alias MLServe.ModelSpec
  alias MLServe.ModelSupervisor
  alias MLServe.Route
  alias MLServe.Telemetry

  @typedoc "A registered model's name."
  @type model :: atom()

  @typedoc "A model version string, for example `\"2.1.0\"`."
  @type version :: String.t()

  @typedoc "Anything the backend accepts as an input."
  @type input :: term()

  @typedoc "Anything the backend produces as a result."
  @type result :: term()

  @typedoc "An error reason returned by MLServe."
  @type reason ::
          :model_not_found
          | :model_not_ready
          | :timeout
          | :overloaded
          | {:invalid_input, term()}
          | {:batch_too_large, pos_integer()}
          | {:backend_error, MLServe.BackendError.t()}
          | {:load_failed, term()}
          | Error.t()

  # Inference

  @doc """
  Runs inference for a single input.

  Executes in the calling process up to the point of dispatch: cache lookup, the `:preprocess`
  hook and admission control all happen here, so a worker is occupied only for actual inference.

  ## Parameters

    - `name`: the registered model name
    - `input`: whatever the backend accepts
    - `opts`: see below

  ## Options

    * `:version` — pin an exact version. Without it, routing follows the default version and any
      active canary.
    * `:timeout` — milliseconds, defaulting to the model's `:timeout`. The deadline travels with
      the request: a worker that picks up already-expired work drops it instead of running it.
      **Not enforced for `concurrency: :shared` backends** — see the note below.
    * `:cache` — `true`, `false`, or a TTL in milliseconds. Off unless asked.
    * `:cache_key` — an explicit cache key. Use it when hashing the full input is more expensive
      than the inference you are skipping.
    * `:cache_ttl` — TTL override in milliseconds.

  > #### Timeouts and shared backends {: .info}
  >
  > A `concurrency: :shared` backend runs `predict/2` **in your own process**, which is what makes
  > it free of message copies. The consequence is that MLServe cannot interrupt it: there is no
  > other process to abandon, and `:timeout` is not enforced. This is the same contract as calling
  > any function directly.
  >
  > If you need MLServe to bound inference time, use `concurrency: :exclusive` so the work happens
  > in a worker the caller can walk away from. Otherwise bound it where the timeout belongs — the
  > backend itself, or the enclosing request.

  ## Examples

      MLServe.predict(:fraud_detection, %{amount: 1500.50})
      #=> {:ok, %{prediction: :fraud, probability: 0.94}}

      MLServe.predict(:fraud_detection, features, version: "2.1.0", timeout: 250)

      MLServe.predict(:embeddings, "some text", cache: true, cache_ttl: :timer.minutes(10))
  """
  @spec predict(model(), input(), keyword()) :: {:ok, result()} | {:error, reason()}
  def predict(name, input, opts \\ []) when is_atom(name) and is_list(opts) do
    Dispatcher.predict(name, input, opts)
  end

  @doc """
  Same as `predict/3` but raises `MLServe.Error` instead of returning `{:error, reason}`.

  ## Examples

      MLServe.predict!(:fraud_detection, features)
      #=> %{prediction: :fraud, probability: 0.94}
  """
  @spec predict!(model(), input(), keyword()) :: result()
  def predict!(name, input, opts \\ []) do
    case predict(name, input, opts) do
      {:ok, result} -> result
      {:error, reason} -> raise Error.wrap(reason, model: name, version: opts[:version])
    end
  end

  @doc """
  Runs inference for a list of inputs, returning results in the same order.

  When the backend implements `c:MLServe.Model.batch_predict/2` this is a **single** backend
  call, which is where the win is — one round-trip and one vectorised computation instead of N.
  Otherwise MLServe maps `c:MLServe.Model.predict/2` over the inputs and stops at the first
  error, since a batch result is all-or-nothing.

  This is for one caller with many inputs. To coalesce *many concurrent callers* into shared
  batches, configure `:batching` instead — see `MLServe.Batcher`.

  ## Parameters

    - `name`: the registered model name
    - `inputs`: a list of inputs
    - `opts`: as `predict/3`, except `:cache`, which does not apply to batches

  ## Examples

      MLServe.batch_predict(:fraud_detection, [%{amount: 10}, %{amount: 20_000}])
      #=> {:ok, [%{prediction: :legitimate, ...}, %{prediction: :fraud, ...}]}
  """
  @spec batch_predict(model(), [input()], keyword()) :: {:ok, [result()]} | {:error, reason()}
  def batch_predict(name, inputs, opts \\ []) when is_atom(name) do
    Dispatcher.batch_predict(name, inputs, opts)
  end

  @doc """
  Same as `batch_predict/3` but raises `MLServe.Error` on failure.
  """
  @spec batch_predict!(model(), [input()], keyword()) :: [result()]
  def batch_predict!(name, inputs, opts \\ []) do
    case batch_predict(name, inputs, opts) do
      {:ok, results} -> results
      {:error, reason} -> raise Error.wrap(reason, model: name, version: opts[:version])
    end
  end

  # Lifecycle

  @doc """
  Registers and loads a model.

  Returns as soon as the model is *registered*; loading proceeds in the background and the model
  reports `:loading` until it is ready. Use `await_ready/2` to block, or `ready?/1` for a
  readiness probe. A load failure is retried with backoff before the model is marked `:failed`.

  Loading a version that is already loaded returns `{:error, {:already_loaded, name, version}}`.
  Loading a *different* version of the same name is how zero-downtime upgrades start: both run
  side by side until you `promote/2`.

  ## Parameters

    - `name`: the atom to register the model under
    - `opts`: the model options documented in `MLServe.Config`

  ## Examples

      MLServe.load_model(:fraud_detection,
        backend: MyApp.FraudModel,
        version: "2.1.0",
        workers: 8,
        config: [threshold: 0.8]
      )
      #=> {:ok, {:fraud_detection, "2.1.0"}}
  """
  @spec load_model(model(), keyword()) :: {:ok, {model(), version()}} | {:error, reason()}
  def load_model(name, opts) when is_atom(name) and is_list(opts) do
    with {:ok, spec} <- Config.build(name, opts),
         {:ok, _route} <- ModelRegistry.register(spec),
         {:ok, _pid} <- start_instance(spec) do
      {:ok, ModelSpec.key(spec)}
    end
  end

  @doc """
  Drains and unloads a model version.

  New requests stop routing to it immediately. Requests already accepted are given up to
  `:drain_timeout` to finish before the workers are terminated, and the number still outstanding
  is reported as the `drained` measurement on `[:ml_serve, :model, :unload]`.

  If the unloaded version was the default and other versions remain, the newest survivor becomes
  the default so unpinned calls keep working.

  ## Parameters

    - `name`: the model name
    - `opts`: `:version` (defaults to the current default version), `:timeout` (drain timeout)

  ## Examples

      MLServe.unload_model(:fraud_detection, version: "1.0.0")
      #=> :ok
  """
  @spec unload_model(model(), keyword()) :: :ok | {:error, reason()}
  def unload_model(name, opts \\ []) when is_atom(name) do
    with {:ok, version} <- resolve_version(name, opts),
         {:ok, spec} <- ModelRegistry.spec(name, version) do
      started_at = System.monotonic_time()
      drain_timeout = Keyword.get(opts, :timeout, spec.drain_timeout)

      drained =
        case ModelServer.drain(name, version, drain_timeout) do
          {:ok, outstanding} -> outstanding
          {:error, _} -> 0
        end

      ModelRegistry.update_status(name, version, :unloading)
      ModelSupervisor.stop_model(name, version)
      ModelRegistry.unregister(name, version)
      MLServe.Cache.invalidate(name, version)

      Telemetry.model_unload(
        %{model: name, version: version, backend: spec.backend},
        System.monotonic_time() - started_at,
        drained
      )

      :ok
    end
  end

  @doc """
  Reloads a model version in place, picking up a changed artifact or configuration.

  Equivalent to `unload_model/2` followed by `load_model/2` with the same options, which means it
  *is* a gap in availability for that version. For a zero-downtime change, load a new version
  alongside and `promote/2` instead.

  ## Parameters

    - `name`: the model name
    - `opts`: `:version`, plus any model options to override
  """
  @spec reload_model(model(), keyword()) :: {:ok, {model(), version()}} | {:error, reason()}
  def reload_model(name, opts \\ []) when is_atom(name) do
    with {:ok, version} <- resolve_version(name, opts),
         {:ok, spec} <- ModelRegistry.spec(name, version) do
      merged = spec |> to_opts() |> Keyword.merge(Keyword.delete(opts, :timeout))

      with :ok <- unload_model(name, Keyword.take(opts, [:version, :timeout])) do
        load_model(name, merged)
      end
    end
  end

  # Introspection

  @doc """
  Lists every registered model name.

  ## Examples

      MLServe.models()
      #=> [:fraud_detection, :recommendations]
  """
  @spec models() :: [model()]
  def models, do: ModelRegistry.list()

  @doc """
  Returns operational status for a model version.

  ## Parameters

    - `name`: the model name
    - `opts`: `:version`, defaulting to the current default version

  ## Examples

      MLServe.model_status(:fraud_detection)
      #=> {:ok, %{
      #=>   name: :fraud_detection,
      #=>   version: "1.0.0",
      #=>   status: :ready,
      #=>   backend: MyApp.FraudModel,
      #=>   workers: 4,
      #=>   concurrency: :exclusive,
      #=>   in_flight: 3,
      #=>   requests: 154223,
      #=>   errors: 12,
      #=>   default?: true,
      #=>   canary: nil,
      #=>   ...
      #=> }}

  `requests`, `errors` and `in_flight` come from atomic counters updated on the hot path — no
  process is asked, and reading them costs nothing.
  """
  @spec model_status(model(), keyword()) :: {:ok, map()} | {:error, :model_not_found}
  def model_status(name, opts \\ []) when is_atom(name) do
    with {:ok, version} <- resolve_version(name, opts),
         {:ok, entry} <- ModelRegistry.status(name, version),
         {:ok, route} <- ModelRegistry.peek_route(name, version) do
      spec = entry.spec

      {:ok,
       %{
         name: name,
         version: version,
         status: entry.status,
         backend: spec.backend,
         concurrency: spec.concurrency,
         workers: ModelSpec.worker_count(spec),
         selection: spec.selection,
         batching: spec.batching,
         native_batching: Backend.supports_batching?(spec.backend),
         cache: spec.cache,
         max_concurrency: spec.max_concurrency,
         timeout: spec.timeout,
         default?: ModelRegistry.default_version(name) == {:ok, version},
         canary: ModelRegistry.canary(name),
         loaded_at: entry.loaded_at,
         load_duration_ms: entry.load_duration_ms,
         in_flight: Route.in_flight(route),
         requests: Route.requests(route),
         errors: Route.errors(route),
         failure: entry.failure,
         metadata: entry.metadata
       }}
    end
  end

  @doc """
  Lists every loaded version of a model, oldest first.

  ## Examples

      MLServe.versions(:fraud_detection)
      #=> {:ok, ["1.0.0", "2.1.0"]}
  """
  @spec versions(model()) :: {:ok, [version()]} | {:error, :model_not_found}
  def versions(name) when is_atom(name), do: ModelRegistry.versions(name)

  @doc """
  Returns true when the model is loaded and serving.

  Accepts a model name, a `{name, version}` tuple to check one exact version, or `:all`.

  With `:all` (the default), returns true only when every registered model is ready — which is
  what a container readiness probe wants:

      # in a Phoenix router
      get "/health/ready", HealthController, :ready

      def ready(conn, _params) do
        if MLServe.ready?(), do: send_resp(conn, 200, "ok"), else: send_resp(conn, 503, "loading")
      end
  """
  @spec ready?(model() | {model(), version()} | :all) :: boolean()
  def ready?(name \\ :all)

  def ready?(:all) do
    case models() do
      [] -> true
      names -> Enum.all?(names, &ready?/1)
    end
  end

  def ready?({name, version}) when is_atom(name) and is_binary(version) do
    match?({:ok, _route}, ModelRegistry.fetch_route(name, version))
  end

  def ready?(name) when is_atom(name) do
    match?({:ok, _route}, ModelRegistry.route(name))
  end

  @doc """
  Blocks until a model is ready, or the timeout expires.

  Models load asynchronously, which makes both tests and startup scripts racy without this.

  Takes the same argument as `ready?/1`, so a specific version can be awaited while an older one
  is still the default — which is exactly the situation during a version rollout.

  ## Examples

      MLServe.load_model(:fraud_detection, backend: MyApp.FraudModel)
      :ok = MLServe.await_ready(:fraud_detection)

      MLServe.load_model(:fraud_detection, backend: MyApp.FraudModel, version: "2.1.0")
      :ok = MLServe.await_ready({:fraud_detection, "2.1.0"})
  """
  @spec await_ready(model() | {model(), version()} | :all, timeout()) :: :ok | {:error, :timeout}
  def await_ready(name \\ :all, timeout \\ 5_000) do
    deadline = System.monotonic_time(:millisecond) + timeout
    do_await_ready(name, deadline)
  end

  # Versioning

  @doc """
  Makes `version` the default for unpinned traffic, and clears any canary.

  The switch is a single ETS write: in-flight requests finish on whatever version they started
  on, and the next request routes to the new one. No restart, no downtime.

  ## Examples

      MLServe.load_model(:fraud_detection, backend: MyApp.FraudModel, version: "2.1.0")
      MLServe.await_ready(:fraud_detection)
      MLServe.promote(:fraud_detection, "2.1.0")
      MLServe.unload_model(:fraud_detection, version: "1.0.0")
  """
  @spec promote(model(), version()) :: :ok | {:error, :model_not_found}
  def promote(name, version) when is_atom(name) and is_binary(version) do
    ModelRegistry.set_default(name, version)
  end

  @doc """
  Routes `percent` of unpinned traffic to `version` for progressive rollout.

  Each request rolls independently in the calling process, so there is no coordination point and
  no shared counter. Telemetry metadata carries both `version` and `canary?`, which is what makes
  the rollout decidable: your metrics backend can compare error rate and latency per version
  without extra instrumentation.

  A candidate that is not ready never black-holes its share — those requests fall back to the
  default version.

  ## Parameters

    - `name`: the model name
    - `version`: the candidate version, already loaded
    - `percent`: 1..100

  ## Examples

      MLServe.canary(:fraud_detection, "2.1.0", 5)   # 5% of traffic
      # ... watch [:ml_serve, :prediction, :stop] grouped by version ...
      MLServe.promote(:fraud_detection, "2.1.0")     # ship it, clears the canary
  """
  @spec canary(model(), version(), 1..100) :: :ok | {:error, term()}
  def canary(name, version, percent)
      when is_atom(name) and is_binary(version) and is_integer(percent) and percent >= 1 and
             percent <= 100 do
    ModelRegistry.set_canary(name, version, percent)
  end

  @doc """
  Aborts an in-progress canary, sending all unpinned traffic back to the default version.
  """
  @spec clear_canary(model()) :: :ok
  def clear_canary(name) when is_atom(name), do: ModelRegistry.clear_canary(name)

  # Private Functions

  defp start_instance(spec) do
    case ModelSupervisor.start_model(spec) do
      {:ok, pid} ->
        {:ok, pid}

      {:error, {:already_started, pid}} ->
        {:ok, pid}

      {:error, reason} ->
        ModelRegistry.unregister(spec.name, spec.version)
        {:error, {:load_failed, reason}}
    end
  end

  defp resolve_version(name, opts) do
    case Keyword.get(opts, :version) do
      nil -> ModelRegistry.default_version(name)
      version -> {:ok, version}
    end
  end

  defp do_await_ready(name, deadline) do
    cond do
      ready?(name) -> :ok
      System.monotonic_time(:millisecond) >= deadline -> {:error, :timeout}
      true -> sleep_and_retry(name, deadline)
    end
  end

  defp sleep_and_retry(name, deadline) do
    Process.sleep(5)
    do_await_ready(name, deadline)
  end

  defp to_opts(%ModelSpec{} = spec) do
    [
      backend: spec.backend,
      version: spec.version,
      path: spec.path,
      checksum: spec.checksum,
      workers: spec.workers,
      concurrency: spec.concurrency,
      load: spec.load,
      timeout: spec.timeout,
      drain_timeout: spec.drain_timeout,
      max_concurrency: spec.max_concurrency,
      max_batch_size: spec.max_batch_size,
      batching: spec.batching && Map.to_list(spec.batching),
      cache: spec.cache && Map.to_list(spec.cache),
      preprocess: spec.preprocess,
      postprocess: spec.postprocess,
      restart_on_error: spec.restart_on_error,
      selection: spec.selection,
      config: Keyword.drop(spec.config, [:version, :path])
    ]
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
  end
end
