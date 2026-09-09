# Quick start — the smallest useful MLServe setup.
#
#     elixir examples/scripts/01_quick_start.exs
#
# Covers: defining a backend, loading it at runtime, single predictions, the bang variant,
# operational status, and unloading.

Mix.install([
  # Running from a clone of the repo. Against the published package this is:
  #   {:ml_serve, "~> 0.1.0"}
  {:ml_serve, path: Path.expand("../..", __DIR__)}
])

section = fn title ->
  IO.puts("\n── " <> title <> " " <> String.duplicate("─", max(0, 62 - String.length(title))))
end

defmodule FraudModel do
  @moduledoc """
  A model is any module with `load/1` and `predict/2`. That is the whole contract.

  There is no ML runtime here — this one is arithmetic — but MLServe cannot tell the difference
  and does not care. The same pooling, batching, caching, telemetry and versioning apply whether
  the thing behind `predict/2` is an `Nx.Serving`, an ONNX session or an `if` statement.
  """
  @behaviour MLServe.Model

  @impl true
  def load(config) do
    # Whatever `load/1` returns becomes the state handed to every `predict/2` call. A real backend
    # opens its ONNX session or builds its Nx.Serving here, so the cost is paid once at load
    # rather than once per request.
    {:ok, %{threshold: Keyword.fetch!(config, :threshold)}}
  end

  @impl true
  def predict(%{threshold: threshold}, features) do
    %{amount: amount} = features
    failed = Map.get(features, :failed_transactions_24h, 0)

    probability =
      (amount / 2_000 + failed * 0.05)
      |> min(1.0)
      |> Float.round(4)

    {:ok,
     %{
       prediction: if(probability > threshold, do: :fraud, else: :legitimate),
       probability: probability
     }}
  end

  # Optional callback. Whatever it returns shows up under :metadata in MLServe.model_status/2,
  # which is a good place to record what the model was trained on.
  @impl true
  def metadata(%{threshold: threshold}) do
    %{threshold: threshold, features: [:amount, :failed_transactions_24h]}
  end
end

section.("Loading a model")

# Models can be declared in config (see examples/inference_service) or loaded at runtime. Both
# go through the same validation and the same code path.
{:ok, {name, version}} =
  MLServe.load_model(:fraud_detection,
    backend: FraudModel,
    version: "1.0.0",
    workers: 4,
    config: [threshold: 0.7]
  )

IO.puts("registered #{inspect(name)} version #{version}")

# load_model/2 returns as soon as the model is *registered*. Loading proceeds in the background,
# so a thirty-second model load never blocks application boot. This is the readiness probe.
:ok = MLServe.await_ready(:fraud_detection)
IO.puts("ready?          #{MLServe.ready?(:fraud_detection)}")

section.("Predicting")

# The everyday call. Returns {:ok, result} or {:error, reason} — never raises for an expected
# failure, so it composes with `with`.
{:ok, result} =
  MLServe.predict(:fraud_detection, %{
    amount: 1500.50,
    transaction_count_24h: 8,
    failed_transactions_24h: 2
  })

IO.inspect(result, label: "flagged")

{:ok, benign} = MLServe.predict(:fraud_detection, %{amount: 12.00})
IO.inspect(benign, label: "cleared")

# The bang variant raises MLServe.Error instead of returning a tuple. Use it where a failure is
# genuinely exceptional and you would only re-raise anyway.
IO.inspect(MLServe.predict!(:fraud_detection, %{amount: 4_000.00}), label: "predict!")

section.("Many inputs at once")

# batch_predict/3 hands the whole list to the backend in one call when it implements
# batch_predict/2, and otherwise maps over it. Either way the caller's code is identical.
{:ok, results} =
  MLServe.batch_predict(:fraud_detection, [
    %{amount: 10.0},
    %{amount: 900.0},
    %{amount: 5_000.0}
  ])

for {amount, result} <- Enum.zip([10.0, 900.0, 5_000.0], results) do
  amount = :erlang.float_to_binary(amount, decimals: 2)
  IO.puts("  #{String.pad_leading(amount, 8)} → #{result.prediction} (#{result.probability})")
end

section.("Operational status")

{:ok, status} = MLServe.model_status(:fraud_detection)

IO.puts("""
  status        #{status.status}
  backend       #{inspect(status.backend)}
  concurrency   #{status.concurrency} across #{status.workers} worker(s)
  requests      #{status.requests} (#{status.errors} errors, #{status.in_flight} in flight)
  loaded in     #{status.load_duration_ms}ms
  default?      #{status.default?}
  metadata      #{inspect(status.metadata)}\
""")

# requests/errors/in_flight come from atomic counters written on the hot path. Reading them asks
# no process anything, so a /health endpoint can call this per request without a second thought.
IO.inspect(MLServe.models(), label: "\nloaded models")

section.("Unloading")

# Waits up to :drain_timeout for in-flight requests to finish before tearing the subtree down.
:ok = MLServe.unload_model(:fraud_detection)

IO.puts(
  "unloaded — predict now returns #{inspect(MLServe.predict(:fraud_detection, %{amount: 1.0}))}"
)
