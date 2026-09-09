defmodule MLServe.Model do
  @moduledoc """
  The behaviour every MLServe model backend implements.

  A backend is the adapter between MLServe's runtime — supervision, pooling, batching, caching,
  telemetry — and an actual inference engine. MLServe never interprets a model artifact itself;
  it hands the configuration to `c:load/1` and calls `c:predict/2` with whatever state you return.

  Only `c:load/1` and `c:predict/2` are required. The smallest useful backend is two functions:

      defmodule MyApp.Backends.Threshold do
        @behaviour MLServe.Model

        @impl true
        def load(config), do: {:ok, Keyword.fetch!(config, :threshold)}

        @impl true
        def predict(threshold, %{score: score}) do
          {:ok, %{fraud?: score > threshold}}
        end
      end

  ## Capabilities

  `c:capabilities/0` tells MLServe *how* the backend may be executed. It is the single most
  important thing a backend declares, because it changes the execution strategy entirely:

  | `:concurrency` | Effect |
  | -------------- | ------ |
  | `:exclusive` (default) | State lives in pooled `MLServe.Worker` processes; one request per worker at a time. Use for anything not safe to call concurrently — ONNX Runtime sessions, ports, stateful handles. |
  | `:shared` | `c:predict/2` runs **in the calling process** and state is read from `:persistent_term`. No worker processes are started and large inputs are never copied between processes. Use for `Nx.Serving`, pure functions, and remote HTTP services. |

  | `:load` | Effect |
  | ------- | ------ |
  | `:once` (default) | `c:load/1` is called once and the resulting state term is shared by every worker. Correct for NIF-resource-backed models, where the term is a cheap handle and the real memory is shared. |
  | `:per_worker` | `c:load/1` is called once per worker, each getting independent state. Correct for ports and per-worker sessions. |

  Choosing `:once` by default matters: loading a 2 GB model separately into eight workers is an
  out-of-memory crash, not a pool.

  ## Optional callbacks

  MLServe detects optional callbacks with `function_exported?/3`, so implementing one is enough —
  there is no registration step.

    * `c:batch_predict/2` — receives a list of inputs and returns a list of results in the same
      order. Implement it whenever the engine can vectorise; both `MLServe.batch_predict/3` and
      dynamic batching route through it. Without it MLServe maps `c:predict/2` over the inputs.
      Return `{:error, :not_supported}` to fall back to that mapping for a particular model.
    * `c:unload/1` — release ports, files, or NIF resources. Called on unload and on reload.
    * `c:metadata/1` — arbitrary map surfaced under `:metadata` in `MLServe.model_status/2`.
      Good place for input shapes, label names, or a training run id.
    * `c:capabilities/0` — as described above.

  ## Error handling

  Return `{:error, reason}` for expected failures. MLServe also traps raises, throws and exits
  around every callback and converts them to `{:error, {:backend_error, %MLServe.BackendError{}}}`
  with the original exception and stacktrace preserved — a backend never needs its own `try`
  purely to protect the runtime.

  See the [Creating a Model Backend](creating-a-backend.md) guide for complete Nx, Bumblebee,
  ONNX and Python-port implementations.
  """

  @typedoc "Backend-private state returned by `c:load/1` and passed to every other callback."
  @type state :: term()

  @typedoc "A single inference input, in whatever shape the backend accepts."
  @type input :: term()

  @typedoc "A single inference result, in whatever shape the backend produces."
  @type result :: term()

  @typedoc "How MLServe may execute this backend."
  @type capabilities :: %{
          optional(:concurrency) => :shared | :exclusive,
          optional(:load) => :once | :per_worker
        }

  @doc """
  Loads the model and returns its state.

  Receives the model's `:config` keyword list, with `:path` (already validated and expanded when
  configured) and `:version` injected. Called once, or once per worker when
  `c:capabilities/0` declares `load: :per_worker`.
  """
  @callback load(config :: keyword()) :: {:ok, state()} | {:error, term()}

  @doc """
  Runs inference for a single input.
  """
  @callback predict(state(), input()) :: {:ok, result()} | {:error, term()}

  @doc """
  Runs inference for a list of inputs, returning results in the same order.

  Optional. When absent MLServe maps `c:predict/2` over the inputs.
  """
  @callback batch_predict(state(), [input()]) :: {:ok, [result()]} | {:error, term()}

  @doc """
  Releases any resources held by the state. Optional.
  """
  @callback unload(state()) :: :ok

  @doc """
  Returns a map of backend-specific information for `MLServe.model_status/2`. Optional.
  """
  @callback metadata(state()) :: map()

  @doc """
  Declares how MLServe may execute this backend. Optional; defaults to
  `%{concurrency: :exclusive, load: :once}`.
  """
  @callback capabilities() :: capabilities()

  @optional_callbacks batch_predict: 2, unload: 1, metadata: 1, capabilities: 0
end
