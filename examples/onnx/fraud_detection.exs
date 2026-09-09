# Serving a real ONNX model.
#
#     elixir examples/onnx/fraud_detection.exs
#
# Unlike the scripts in examples/scripts, this one loads an actual model file through ONNX
# Runtime. Everything below is genuinely running inference — there is no arithmetic stand-in.
#
# See transformer.exs for the same treatment of a model with far more awkward constraints.
#
# The first run compiles Ortex's Rust NIF and downloads ONNX Runtime, which takes a few minutes.
# Later runs start immediately.
#
# Covers the parts of MLServe that only matter once a model is a *file* on disk: path containment
# against :model_root, checksum verification, one session shared across a worker pool, and real
# batched inference through a single Ortex.run/2.

Mix.install([
  # Running from a clone of the repo. Against the published package this is:
  #   {:ml_serve, "~> 0.1.0"}
  {:ml_serve, path: Path.expand("../..", __DIR__)},
  {:ortex, "~> 0.1.10"},
  {:nx, "~> 0.7"}
])

section = fn title ->
  IO.puts("\n── " <> title <> " " <> String.duplicate("─", max(0, 62 - String.length(title))))
end

# Model artifacts are resolved relative to :model_root and may not escape it. Point it at this
# directory; a real deployment uses priv/models or a volume mounted at a fixed path.
Application.put_env(:ml_serve, :model_root, __DIR__)

# sha256 of fraud_detection.onnx. Recompute with MLServe.Security.digest(path, :sha256).
checksum = "fc33dfb1f75e1deb5bb27dc6c77886a0785522daa997fba11f4b28df6994ce4d"

defmodule FraudONNX do
  @moduledoc """
  An ONNX Runtime session behind the MLServe.Model behaviour.

  The session handle is a NIF resource: cheap to copy, backed by memory the runtime owns. That is
  what makes load: :once correct here — all four workers share one session and therefore one copy
  of the weights. load: :per_worker would multiply that memory by the pool size, which on a GPU is
  the difference between fitting and not.

  concurrency: :exclusive is the conservative choice: ONNX Runtime sessions are not guaranteed
  safe to call concurrently, so MLServe serialises access through a pool rather than running
  predict/2 in every caller at once.
  """

  @behaviour MLServe.Model

  # Raw features are on wildly different scales — currency, counts, counts. The model was trained
  # on normalised inputs, so the backend owns that normalisation. Doing it here rather than at the
  # call site means every caller cannot get it subtly wrong in its own way.
  @scale %{amount: 10_000, transaction_count_24h: 100, failed_transactions_24h: 10}
  @features [:amount, :transaction_count_24h, :failed_transactions_24h]

  @impl true
  def capabilities, do: %{concurrency: :exclusive, load: :once}

  @impl true
  def load(config) do
    # MLServe has already resolved this path, confirmed it sits inside :model_root, followed any
    # symlinks, checked it is a readable regular file within :max_model_bytes, and verified the
    # checksum. By the time load/1 runs, the only open question is whether ONNX Runtime likes it.
    path = Keyword.fetch!(config, :path)

    IO.puts("    [load/1 ran] building the ONNX session from #{Path.basename(path)}")

    {:ok, %{session: Ortex.load(path)}}
  rescue
    error -> {:error, {:ortex_load_failed, Exception.message(error)}}
  end

  @impl true
  def predict(state, features) do
    tensor = Nx.tensor([encode(features)], type: :f32)
    {output} = Ortex.run(state.session, {tensor})

    {:ok, output |> Nx.to_flat_list() |> hd() |> decode()}
  end

  # The whole batch becomes one tensor and one Ortex.run/2. This is what MLServe's :batching
  # window feeds, and why the window is worth having: the fixed cost of crossing into the runtime
  # is paid once for the batch rather than once per row.
  @impl true
  def batch_predict(state, batch) do
    tensor = batch |> Enum.map(&encode/1) |> Nx.tensor(type: :f32)
    {output} = Ortex.run(state.session, {tensor})

    {:ok, output |> Nx.to_flat_list() |> Enum.map(&decode/1)}
  end

  @impl true
  def metadata(_state), do: %{features: @features, runtime: "onnxruntime via ortex"}

  defp encode(features) do
    Enum.map(@features, fn feature ->
      Map.get(features, feature, 0) / Map.fetch!(@scale, feature)
    end)
  end

  defp decode(probability) do
    %{
      prediction: if(probability > 0.5, do: :fraud, else: :legitimate),
      probability: Float.round(probability, 4)
    }
  end
end

section.("Loading a model from disk")

{:ok, _} =
  MLServe.load_model(:fraud_detection,
    backend: FraudONNX,
    version: "1.0.0",
    # Relative to :model_root. An absolute path pointing outside it is rejected, and so is any
    # amount of ../ — see the last section.
    path: "fraud_detection.onnx",
    # Integrity: the file is hashed before the backend is handed the path, so an artifact that
    # changed in transit is never loaded.
    checksum: {:sha256, checksum},
    workers: 4,
    batching: [max_size: 32, timeout: 10]
  )

:ok = MLServe.await_ready(:fraud_detection)

{:ok, status} = MLServe.model_status(:fraud_detection)

IO.puts("""
    status        #{status.status}
    workers       #{status.workers} (#{status.concurrency})
    loaded in     #{status.load_duration_ms}ms
    metadata      #{inspect(status.metadata)}

    load/1 ran once, not four times: load: :once means one session is shared by the whole pool.
    On a GPU that is the difference between one copy of the weights and four.\
""")

section.("Real inference")

transactions = [
  %{label: "small purchase", amount: 50.0, transaction_count_24h: 3, failed_transactions_24h: 0},
  %{label: "unusual", amount: 1_500.0, transaction_count_24h: 8, failed_transactions_24h: 2},
  %{label: "card testing", amount: 9_000.0, transaction_count_24h: 30, failed_transactions_24h: 5}
]

for %{label: label} = transaction <- transactions do
  {:ok, result} = MLServe.predict(:fraud_detection, Map.delete(transaction, :label))

  probability = :erlang.float_to_binary(result.probability, decimals: 4)

  IO.puts(
    "    #{String.pad_trailing(label, 16)} #{String.pad_leading(probability, 8)}  #{result.prediction}"
  )
end

section.("Batched inference: one Ortex.run for the whole list")

features = Enum.map(transactions, &Map.delete(&1, :label))
{:ok, results} = MLServe.batch_predict(:fraud_detection, features)

IO.puts("    #{length(results)} rows through a single [#{length(results)}, 3] tensor")
IO.puts("    probabilities  #{inspect(Enum.map(results, & &1.probability))}")

section.("The checksum is enforced, not decorative")

# The same file, one character of the expected digest changed. MLServe hashes the artifact before
# the backend is given the path, so a corrupted or swapped model never reaches ONNX Runtime.
wrong = String.replace_prefix(checksum, String.first(checksum), "0")

{:ok, _} =
  MLServe.load_model(:tampered,
    backend: FraudONNX,
    path: "fraud_detection.onnx",
    checksum: {:sha256, wrong}
  )

# Long enough for the retry budget to be exhausted.
Process.sleep(1_200)
{:ok, tampered} = MLServe.model_status(:tampered)

IO.puts("    status        #{tampered.status}")
IO.puts("    failure       #{inspect(tampered.failure)}")
IO.puts("    predict       #{inspect(MLServe.predict(:tampered, %{amount: 1.0}))}")

IO.puts(
  "\n    Note load/1 never ran for :tampered — the file was rejected before the backend saw it."
)

section.("A model path cannot escape :model_root")

# :path is validated by MLServe.Security before any backend sees it. A model file is data, never
# code, and the path naming it is attacker-influenced in more deployments than people think.
for candidate <- ["fraud_detection.onnx", "../../etc/passwd", "/etc/hosts", "missing.onnx"] do
  outcome =
    case MLServe.Security.validate_path(candidate, root: __DIR__) do
      {:ok, resolved} -> "accepted — " <> Path.basename(resolved)
      {:error, {:invalid_path, reason}} -> "rejected — #{reason}"
    end

  IO.puts("    #{String.pad_trailing(candidate, 22)} #{outcome}")
end

for model <- MLServe.models(), do: MLServe.unload_model(model)
