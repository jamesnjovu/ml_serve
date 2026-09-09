defmodule InferenceService.Models.Fraud do
  @moduledoc """
  A scoring model with a worker pool.

  Declares `concurrency: :exclusive`, which is what you want when the real backend is an ONNX
  session, a port to a Python process, or anything else that is not safe to call concurrently.
  `load: :once` means the state below is built a single time and shared by all four workers,
  rather than once per worker — the right choice for a NIF resource or a large read-only table.
  """

  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :exclusive, load: :once}

  @impl true
  def load(config) do
    threshold = Keyword.get(config, :threshold, 0.7)

    if threshold < 0 or threshold > 1 do
      # Returning an error here is what triggers MLServe's backoff-and-retry, and eventually
      # marks the model :failed with this reason preserved in MLServe.model_status/2.
      {:error, {:invalid_threshold, threshold}}
    else
      {:ok, %{threshold: threshold, weights: %{amount: 1 / 2_000, failed: 0.05, velocity: 0.005}}}
    end
  end

  @impl true
  def predict(%{threshold: threshold, weights: weights}, features) do
    with {:ok, amount} <- fetch_number(features, "amount") do
      failed = number(features, "failed_transactions_24h", 0)
      velocity = number(features, "transaction_count_24h", 0)

      probability =
        (amount * weights.amount + failed * weights.failed + velocity * weights.velocity)
        |> min(1.0)
        |> Float.round(4)

      {:ok,
       %{
         prediction: if(probability > threshold, do: "fraud", else: "legitimate"),
         probability: probability,
         threshold: threshold
       }}
    end
  end

  @impl true
  def metadata(%{threshold: threshold}) do
    %{
      threshold: threshold,
      features: ["amount", "failed_transactions_24h", "transaction_count_24h"]
    }
  end

  defp fetch_number(features, key) do
    case Map.get(features, key) do
      value when is_number(value) -> {:ok, value}
      nil -> {:error, {:missing_feature, key}}
      other -> {:error, {:invalid_feature, key, other}}
    end
  end

  defp number(features, key, default) do
    case Map.get(features, key) do
      value when is_number(value) -> value
      _ -> default
    end
  end
end
