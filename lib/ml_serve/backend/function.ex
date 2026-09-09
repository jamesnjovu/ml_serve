defmodule MLServe.Backend.Function do
  @moduledoc """
  Serves an ordinary Elixir function as a model.

  The simplest backend there is, and more useful than it looks. Rules engines, heuristics,
  score thresholds and glue around a remote service are all just functions, and wrapping one in
  MLServe gives it the same pooling, batching, caching, telemetry and versioning as a neural
  network — with no ML runtime involved.

  It is also the backend the test suite runs on, which is deliberate: MLServe's own tests must
  not need a multi-gigabyte model or a Rust toolchain.

  ## Configuration

      config :ml_serve,
        models: [
          risk_score: [
            backend: MLServe.Backend.Function,
            config: [
              predict: fn %{amount: amount} -> {:ok, %{risky?: amount > 10_000}} end
            ]
          ]
        ]

  ### Options

    * `:predict` — required. A 1-arity function, or `{module, function, extra_args}` receiving the
      input as the first argument. Returning a bare value is fine; it is wrapped in `{:ok, value}`.
    * `:batch_predict` — optional. A 1-arity function over the *list* of inputs, returning a list
      of results. Supply it to make `MLServe.batch_predict/3` a single call.
    * `:init` — optional. A 0-arity function run once at load, whose result is passed to
      `:predict` as `{state, input}` instead of a bare input. Use it to build a lookup table.

  ## Concurrency

  Declares `concurrency: :shared`, so predictions run **in the calling process** and no worker
  processes are started. A plain function has no session to serialise access to, and routing it
  through a pool would add message copies and a bottleneck for nothing.

  ## Examples

      iex> {:ok, state} = MLServe.Backend.Function.load(predict: &(&1 * 2))
      iex> MLServe.Backend.Function.predict(state, 21)
      {:ok, 42}
  """

  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(config) do
    with {:ok, predict} <- fetch_fun(config, :predict, required: true),
         {:ok, batch} <- fetch_fun(config, :batch_predict, required: false),
         {:ok, init} <- fetch_fun(config, :init, required: false, arity: 0) do
      state = if init, do: apply_fun(init, []), else: nil
      {:ok, %{predict: predict, batch_predict: batch, state: state, stateful?: init != nil}}
    end
  end

  @impl true
  def predict(%{predict: predict, state: state, stateful?: stateful?}, input) do
    argument = if stateful?, do: {state, input}, else: input

    predict
    |> apply_fun([argument])
    |> normalize()
  end

  @impl true
  def batch_predict(%{batch_predict: nil}, _inputs), do: {:error, :not_supported}

  def batch_predict(%{batch_predict: batch, state: state, stateful?: stateful?}, inputs) do
    argument = if stateful?, do: {state, inputs}, else: inputs

    case apply_fun(batch, [argument]) do
      {:ok, results} when is_list(results) -> {:ok, results}
      {:error, _} = error -> error
      results when is_list(results) -> {:ok, results}
      other -> {:error, {:invalid_return, other}}
    end
  end

  @impl true
  def metadata(%{batch_predict: batch}), do: %{native_batching: batch != nil}

  # Private Functions

  defp fetch_fun(config, key, opts) do
    arity = Keyword.get(opts, :arity, 1)

    case Keyword.get(config, key) do
      nil ->
        if Keyword.get(opts, :required, false) do
          {:error, "MLServe.Backend.Function requires a #{inspect(key)} function in :config"}
        else
          {:ok, nil}
        end

      fun when is_function(fun, arity) ->
        {:ok, fun}

      {mod, fun, args} when is_atom(mod) and is_atom(fun) and is_list(args) ->
        {:ok, {mod, fun, args}}

      other ->
        {:error,
         "#{inspect(key)} must be a #{arity}-arity function or {module, function, args} tuple, " <>
           "got: #{inspect(other)}"}
    end
  end

  defp apply_fun({mod, fun, args}, prepend), do: apply(mod, fun, prepend ++ args)
  defp apply_fun(fun, args) when is_function(fun), do: apply(fun, args)

  defp normalize({:ok, _} = ok), do: ok
  defp normalize({:error, _} = error), do: error
  defp normalize(other), do: {:ok, other}
end
