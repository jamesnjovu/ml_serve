defmodule MLServe.Backend do
  @moduledoc """
  Guarded invocation of `MLServe.Model` callbacks.

  Every call into a backend goes through this module. It exists for three reasons:

    1. **Nothing raises out of a backend into MLServe's runtime.** A raise, throw or exit is
       converted to `{:error, {:backend_error, %MLServe.BackendError{}}}` carrying the original
       exception *and stacktrace*. Errors are surfaced, never swallowed — the point of catching
       is to attach context, not to hide the failure.
    2. **Optional callbacks are resolved once.** `c:MLServe.Model.batch_predict/2` and friends are
       detected with `function_exported?/3` and given sane fallbacks.
    3. **Return values are validated.** A backend that returns a bare value instead of an
       `{:ok, term}` tuple gets a clear error naming the callback, rather than a confusing
       mismatch three layers up.

  Application code does not normally call this module; it is public because backend authors and
  the guides refer to its semantics.
  """

  alias MLServe.BackendError
  alias MLServe.ModelSpec

  @type invocation_result :: {:ok, term()} | {:error, term()}

  @doc """
  Calls `c:MLServe.Model.load/1`.

  The spec's `:config` already carries `:version`, and `:path` when one is configured.
  """
  @spec load(ModelSpec.t()) :: {:ok, term()} | {:error, term()}
  def load(%ModelSpec{backend: backend, config: config} = spec) do
    guard(spec, {:load, 1}, fn -> backend.load(config) end)
  end

  @doc """
  Calls `c:MLServe.Model.predict/2`.
  """
  @spec predict(ModelSpec.t(), term(), term()) :: invocation_result()
  def predict(%ModelSpec{backend: backend} = spec, state, input) do
    guard(spec, {:predict, 2}, fn -> backend.predict(state, input) end)
  end

  @doc """
  Calls `c:MLServe.Model.batch_predict/2`, falling back to a mapped `c:MLServe.Model.predict/2`.

  The fallback stops at the first error rather than running the remaining inputs, since a batch
  result is all-or-nothing. Backends that can vectorise should implement `batch_predict/2`; the
  whole point of `MLServe.batch_predict/3` is one backend round-trip instead of N.

  A backend that exports `batch_predict/2` but cannot batch *this particular model* returns
  `{:error, :not_supported}` to opt back into the mapped fallback.
  """
  @spec batch_predict(ModelSpec.t(), term(), [term()]) :: {:ok, [term()]} | {:error, term()}
  def batch_predict(%ModelSpec{backend: backend} = spec, state, inputs) when is_list(inputs) do
    if function_exported?(backend, :batch_predict, 2) do
      case guard(spec, {:batch_predict, 2}, fn -> backend.batch_predict(state, inputs) end) do
        # A backend may export batch_predict/2 and still be unable to batch a particular model —
        # MLServe.Backend.Function only batches when given a :batch_predict function. Returning
        # :not_supported opts back into the mapped fallback instead of failing the request.
        {:error, :not_supported} -> map_predict(spec, state, inputs)
        result -> validate_batch(result, spec, length(inputs))
      end
    else
      map_predict(spec, state, inputs)
    end
  end

  @doc """
  Calls `c:MLServe.Model.unload/1` when exported. Always returns `:ok`.

  Unload failures are logged, not propagated: the model is going away regardless, and refusing to
  finish an unload because a backend's cleanup raised would leak the whole model instance.
  """
  @spec unload(ModelSpec.t(), term()) :: :ok
  def unload(%ModelSpec{backend: backend} = spec, state) do
    if function_exported?(backend, :unload, 1) do
      case guard(spec, {:unload, 1}, fn -> backend.unload(state) end) do
        {:error, {:backend_error, error}} ->
          require Logger

          Logger.warning(fn ->
            "[ml_serve] #{inspect(spec.name)} backend unload failed: #{Exception.message(error)}"
          end)

          :ok

        _ ->
          :ok
      end
    else
      :ok
    end
  end

  @doc """
  Calls `c:MLServe.Model.metadata/1` when exported, returning `%{}` otherwise.
  """
  @spec metadata(ModelSpec.t(), term()) :: map()
  def metadata(%ModelSpec{backend: backend} = spec, state) do
    if function_exported?(backend, :metadata, 1) do
      case guard(spec, {:metadata, 1}, fn -> {:ok, backend.metadata(state)} end) do
        {:ok, metadata} when is_map(metadata) -> metadata
        _ -> %{}
      end
    else
      %{}
    end
  end

  @doc """
  Returns whether the backend exports `c:MLServe.Model.batch_predict/2`.

  Surfaced in `MLServe.model_status/2` so operators can see whether a batch call is one backend
  round-trip or N.

  ## Examples

      iex> MLServe.Backend.supports_batching?(MLServe.Backend.Static)
      true
  """
  @spec supports_batching?(module()) :: boolean()
  def supports_batching?(backend) when is_atom(backend) do
    Code.ensure_loaded?(backend) and function_exported?(backend, :batch_predict, 2)
  end

  # Private Functions

  defp map_predict(spec, state, inputs) do
    Enum.reduce_while(inputs, {:ok, []}, fn input, {:ok, acc} ->
      case predict(spec, state, input) do
        {:ok, result} -> {:cont, {:ok, [result | acc]}}
        {:error, _} = error -> {:halt, error}
      end
    end)
    |> case do
      {:ok, results} -> {:ok, Enum.reverse(results)}
      error -> error
    end
  end

  defp validate_batch({:ok, results}, _spec, expected) when is_list(results) do
    if length(results) == expected do
      {:ok, results}
    else
      {:error,
       {:backend_error,
        BackendError.new(
          backend: nil,
          callback: {:batch_predict, 2},
          kind: :error,
          reason: %ArgumentError{
            message:
              "batch_predict/2 returned #{length(results)} results for #{expected} inputs; " <>
                "results must be returned in the same order and length as the inputs"
          },
          stacktrace: []
        )}}
    end
  end

  defp validate_batch({:ok, other}, spec, _expected) do
    bad_return(spec, {:batch_predict, 2}, {:ok, other}, "a list of results")
  end

  defp validate_batch(other, _spec, _expected), do: other

  # A backend callback runs arbitrary third-party code — frequently a NIF. Catching :error,
  # :throw and :exit alike is deliberate: a port that dies takes the caller with it via exit,
  # and that must become an error tuple rather than an unexplained worker crash.
  defp guard(spec, callback, fun) do
    case fun.() do
      {:ok, _} = ok -> ok
      {:error, _} = error -> error
      :ok -> {:ok, :ok}
      other -> bad_return(spec, callback, other, "{:ok, result} or {:error, reason}")
    end
  catch
    kind, reason ->
      {:error,
       {:backend_error,
        BackendError.new(
          backend: spec.backend,
          callback: callback,
          kind: kind,
          reason: normalize(kind, reason, __STACKTRACE__),
          stacktrace: __STACKTRACE__,
          model: spec.name,
          version: spec.version
        )}}
  end

  defp normalize(:error, reason, stacktrace), do: Exception.normalize(:error, reason, stacktrace)
  defp normalize(_kind, reason, _stacktrace), do: reason

  defp bad_return(spec, {fun, arity} = callback, got, expected) do
    {:error,
     {:backend_error,
      BackendError.new(
        backend: spec.backend,
        callback: callback,
        kind: :error,
        reason: %ArgumentError{
          message:
            "#{inspect(spec.backend)}.#{fun}/#{arity} must return #{expected}, got: #{inspect(got)}"
        },
        stacktrace: [],
        model: spec.name,
        version: spec.version
      )}}
  end
end
