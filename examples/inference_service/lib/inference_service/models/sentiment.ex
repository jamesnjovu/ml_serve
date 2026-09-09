defmodule InferenceService.Models.Sentiment do
  @moduledoc """
  A batching model that runs in the caller.

  Declares `concurrency: :shared`, so predictions execute in the calling process with state read
  from `:persistent_term` — no worker processes, no message copies. This is the shape an
  `Nx.Serving` or a Bumblebee pipeline wants.

  It also implements the optional `batch_predict/2`, which is what makes the `:batching` window
  in `config/config.exs` worth having: sixteen concurrent HTTP requests become one backend call.
  """

  @behaviour MLServe.Model

  @positive ~w(good great excellent love wonderful best amazing happy)
  @negative ~w(bad terrible awful hate worst broken angry sad)

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(_config),
    do: {:ok, %{positive: MapSet.new(@positive), negative: MapSet.new(@negative)}}

  # Accepts either a decoded JSON object or a bare string, so the router can hand the request
  # body straight through without knowing anything about this model.
  @impl true
  def predict(state, %{"text" => text}) when is_binary(text), do: {:ok, score(state, text)}
  def predict(state, text) when is_binary(text), do: {:ok, score(state, text)}
  def predict(_state, other), do: {:error, {:invalid_feature, "text", other}}

  # One invocation over the whole list. A real backend amortises its fixed cost here — the NIF
  # boundary, the host-to-device copy, the HTTP round trip — which is the entire point.
  @impl true
  def batch_predict(state, inputs) do
    texts = Enum.map(inputs, &text/1)

    if Enum.all?(texts, &is_binary/1) do
      {:ok, Enum.map(texts, &score(state, &1))}
    else
      {:error, {:invalid_feature, "text", "expected strings or objects with a \"text\" key"}}
    end
  end

  defp text(%{"text" => text}), do: text
  defp text(text), do: text

  defp score(state, text) do
    words = text |> String.downcase() |> String.split(~r/\W+/, trim: true)

    positive = Enum.count(words, &MapSet.member?(state.positive, &1))
    negative = Enum.count(words, &MapSet.member?(state.negative, &1))

    {label, score} =
      cond do
        positive > negative -> {"positive", positive / max(positive + negative, 1)}
        negative > positive -> {"negative", negative / max(positive + negative, 1)}
        true -> {"neutral", 0.5}
      end

    %{label: label, score: Float.round(score, 4), tokens: length(words)}
  end
end
