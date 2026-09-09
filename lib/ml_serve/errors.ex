defmodule MLServe.BackendError do
  @moduledoc """
  Raised when a backend callback raises, throws, or exits.

  MLServe wraps every backend invocation. When the backend fails in a way it did not report as
  `{:error, reason}`, the original `kind`, `reason` and `stacktrace` are captured here rather than
  discarded, and the whole struct is returned as `{:error, {:backend_error, %MLServe.BackendError{}}}`.

  Keeping the stacktrace is the point: a model that raises `ArgumentError` deep inside a tensor
  operation is a bug you need the line number for, and a `{:error, :something_went_wrong}` atom
  would throw that away.

  ## Fields

    * `:backend` — the backend module that failed
    * `:callback` — the callback that failed, e.g. `{:predict, 2}`
    * `:kind` — `:error`, `:throw` or `:exit`
    * `:reason` — the raised exception, thrown value, or exit reason
    * `:stacktrace` — the stacktrace at the point of failure
    * `:model` / `:version` — which model was being served
  """

  @type t :: %__MODULE__{
          backend: module() | nil,
          callback: {atom(), arity()} | nil,
          kind: :error | :throw | :exit,
          reason: term(),
          stacktrace: Exception.stacktrace(),
          model: atom() | nil,
          version: String.t() | nil,
          message: String.t()
        }

  defexception [
    :backend,
    :callback,
    :kind,
    :reason,
    :model,
    :version,
    :message,
    stacktrace: []
  ]

  @doc false
  @spec new(keyword()) :: t()
  def new(opts) do
    struct = struct!(__MODULE__, opts)
    %{struct | message: build_message(struct)}
  end

  @impl true
  def message(%__MODULE__{message: nil} = error), do: build_message(error)
  def message(%__MODULE__{message: message}), do: message

  @impl true
  def blame(%__MODULE__{} = error, stacktrace), do: {error, stacktrace}

  defp build_message(%__MODULE__{} = error) do
    "#{describe_callback(error)} failed: " <>
      Exception.format_banner(error.kind, error.reason, [])
  end

  defp describe_callback(%__MODULE__{backend: nil}), do: "backend"

  defp describe_callback(%__MODULE__{backend: backend, callback: nil}) do
    inspect(backend)
  end

  defp describe_callback(%__MODULE__{backend: backend, callback: {fun, arity}}) do
    "#{inspect(backend)}.#{fun}/#{arity}"
  end
end

defmodule MLServe.Error do
  @moduledoc """
  The structured error MLServe raises and reports.

  Every non-bang MLServe function returns `{:error, reason}` where `reason` is a plain atom or
  tagged tuple, as documented on `MLServe.predict/3`. This struct is the *raised* form: bang
  functions such as `MLServe.predict!/3` convert the reason with `wrap/2` and raise it.

  It is an exception, so `raise`, `Exception.message/1` and `rescue MLServe.Error` all work:

      iex> MLServe.Error.wrap(:model_not_found, model: :fraud) |> Exception.message()
      "model :fraud is not loaded"

  ## Fields

    * `:type` — the machine-readable classification, e.g. `:model_not_found`, `:timeout`
    * `:message` — human-readable description
    * `:code` — optional short code carried through from a backend
    * `:details` — optional map of context (model name, version, limits)
  """

  @typedoc "Machine-readable error classification."
  @type type ::
          :model_not_found
          | :model_not_ready
          | :timeout
          | :overloaded
          | :invalid_input
          | :batch_too_large
          | :load_failed
          | :backend_error
          | :config
          | :unknown

  @type t :: %__MODULE__{
          type: type(),
          message: String.t(),
          code: String.t() | nil,
          details: map() | nil
        }

  defexception [:type, :message, :code, :details]

  @retryable [:timeout, :overloaded, :model_not_ready]

  @doc """
  Builds an error struct.

  ## Parameters

    - `type`: the classification atom
    - `message`: human-readable description
    - `opts`: `:code` and `:details`

  ## Examples

      iex> err = MLServe.Error.new(:timeout, "inference timed out", details: %{model: :fraud})
      iex> err.type
      :timeout
  """
  @spec new(type(), String.t(), keyword()) :: t()
  def new(type, message, opts \\ []) do
    %__MODULE__{
      type: type,
      message: message,
      code: Keyword.get(opts, :code),
      details: Keyword.get(opts, :details)
    }
  end

  @doc """
  Converts an `{:error, reason}` reason term into an `MLServe.Error`.

  Used by the bang functions. `details` are merged into the resulting struct so the raised
  message can name the model and version that failed.

  ## Parameters

    - `reason`: the reason term from a non-bang MLServe function
    - `details`: keyword list of context, typically `model:` and `version:`

  ## Examples

      iex> MLServe.Error.wrap(:model_not_ready, model: :fraud).type
      :model_not_ready

      iex> MLServe.Error.wrap({:invalid_input, "amount must be a number"}) |> Exception.message()
      "invalid input: amount must be a number"
  """
  @spec wrap(term(), keyword()) :: t() | MLServe.BackendError.t()
  def wrap(reason, details \\ [])

  def wrap(%__MODULE__{} = error, details), do: merge_details(error, details)
  def wrap(%MLServe.BackendError{} = error, _details), do: error

  def wrap({:backend_error, %MLServe.BackendError{} = error}, _details), do: error

  def wrap(:model_not_found, details) do
    new(:model_not_found, "model #{name(details)} is not loaded", details: map(details))
  end

  def wrap(:model_not_ready, details) do
    new(
      :model_not_ready,
      "model #{name(details)} is not ready to serve requests",
      details: map(details)
    )
  end

  def wrap(:timeout, details) do
    new(:timeout, "inference for model #{name(details)} timed out", details: map(details))
  end

  def wrap(:overloaded, details) do
    new(
      :overloaded,
      "model #{name(details)} is at its concurrency limit",
      details: map(details)
    )
  end

  def wrap({:invalid_input, reason}, details) do
    new(:invalid_input, "invalid input: #{describe(reason)}", details: map(details))
  end

  def wrap({:batch_too_large, max}, details) do
    new(
      :batch_too_large,
      "batch exceeds the maximum size of #{max}",
      details: details |> Map.new() |> Map.put(:max_batch_size, max)
    )
  end

  def wrap({:load_failed, reason}, details) do
    new(
      :load_failed,
      "model #{name(details)} failed to load: #{describe(reason)}",
      details: map(details)
    )
  end

  def wrap({:backend_error, reason}, details) do
    new(:backend_error, "backend error: #{describe(reason)}", details: map(details))
  end

  def wrap(reason, details) do
    new(:unknown, describe(reason), details: map(details))
  end

  @doc """
  Returns true when retrying the same call could plausibly succeed.

  Transient conditions — a timeout, a full queue, a model still loading — are retryable. A missing
  model, invalid input, or a backend that raised are not: retrying re-runs the same failure.

  This is what the [Oban guide](oban-integration.md) uses to choose between `:snooze` and
  `:discard`.

  ## Examples

      iex> MLServe.Error.retryable?(MLServe.Error.wrap(:timeout))
      true

      iex> MLServe.Error.retryable?(MLServe.Error.wrap({:invalid_input, "bad"}))
      false
  """
  @spec retryable?(t() | MLServe.BackendError.t() | term()) :: boolean()
  def retryable?(%__MODULE__{type: type}), do: type in @retryable
  def retryable?(%MLServe.BackendError{}), do: false
  def retryable?(reason), do: reason |> wrap() |> retryable?()

  @impl true
  def message(%__MODULE__{message: message}), do: message

  # Private Functions

  defp name(details) do
    case Keyword.get(details, :model) do
      nil -> "(unknown)"
      model -> inspect(model)
    end
  end

  defp map([]), do: nil
  defp map(details), do: Map.new(details)

  defp merge_details(%__MODULE__{} = error, []), do: error

  defp merge_details(%__MODULE__{details: existing} = error, details) do
    %{error | details: Map.merge(Map.new(details), existing || %{})}
  end

  defp describe(reason) when is_binary(reason), do: reason
  defp describe(reason) when is_atom(reason), do: inspect(reason)
  defp describe(reason), do: inspect(reason)
end
