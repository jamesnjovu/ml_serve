defmodule MLServe.Backend.Static do
  @moduledoc """
  Always returns the same configured result, whatever the input.

  Exists so that applications depending on MLServe can test their own code without a real model.
  Swap the backend in `config/test.exs` and every `MLServe.predict/3` call in the system under
  test returns a known value, with the real routing, caching and telemetry still exercised:

      # config/test.exs
      config :ml_serve,
        models: [
          fraud_detection: [
            backend: MLServe.Backend.Static,
            config: [result: %{prediction: :fraud, probability: 0.94}]
          ]
        ]

  ### Options

    * `:result` — the value returned by every prediction. Defaults to `%{}`.
    * `:error` — when set, every prediction returns `{:error, value}` instead. Useful for
      exercising your application's error path.
    * `:delay` — milliseconds to sleep before returning, for testing timeouts.

  ## Examples

      iex> {:ok, state} = MLServe.Backend.Static.load(result: %{score: 1})
      iex> MLServe.Backend.Static.predict(state, :anything)
      {:ok, %{score: 1}}

      iex> {:ok, state} = MLServe.Backend.Static.load(error: :unavailable)
      iex> MLServe.Backend.Static.predict(state, :anything)
      {:error, :unavailable}
  """

  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(config) do
    {:ok,
     %{
       result: Keyword.get(config, :result, %{}),
       error: Keyword.get(config, :error),
       delay: Keyword.get(config, :delay, 0)
     }}
  end

  @impl true
  def predict(state, _input) do
    if state.delay > 0, do: Process.sleep(state.delay)

    case state.error do
      nil -> {:ok, state.result}
      error -> {:error, error}
    end
  end

  @impl true
  def batch_predict(state, inputs) do
    case predict(state, nil) do
      {:ok, result} -> {:ok, List.duplicate(result, length(inputs))}
      {:error, _} = error -> error
    end
  end

  @impl true
  def metadata(state), do: %{static: true, result: state.result}
end
