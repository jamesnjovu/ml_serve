# Serving a transformer: what changes when the model has awkward constraints.
#
#     elixir examples/onnx/transformer.exs
#
# The companion to fraud_detection.exs. That model is convenient — a dynamic batch dimension, one
# input, one output. This one is a real PyTorch-exported GPT-NeoX with the constraints real
# exports actually have: a batch dimension pinned to 1, a sequence length pinned to 128, two
# inputs of different dtypes, and eleven outputs.
#
# Those constraints are the point. They decide what the backend may declare, and MLServe's job is
# to take that declaration seriously rather than assume every model can batch.
#
# The first run compiles Ortex's Rust NIF and downloads ONNX Runtime, which takes a few minutes.
# Later runs start immediately.

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

Application.put_env(:ml_serve, :model_root, __DIR__)

defmodule TinyGPTNeoX do
  @moduledoc """
  A GPT-NeoX export behind the MLServe.Model behaviour.

  > #### The weights are random {: .warning}
  >
  > gptneox_Opset18.onnx is a shape-and-opset conformance model: correct architecture, untrained
  > weights, a 32-token vocabulary. Every logit below is real arithmetic over real weights, and
  > every one of them is meaningless as language. It is here to exercise MLServe against a model
  > with a transformer's *shape*, not to say anything.

  ## What the export dictates

      input_ids        int64    [1, 128]
      attention_mask   float32  [1, 128]
      logits           float32  [1, 128, 32]
      key/value × 5    float32  [1, 4, 128, 8]

  Both leading dimensions are literal, not symbolic. That is the whole story of this example:

    * **Batch pinned to 1** — the runtime physically cannot take two rows, so this backend does
      not implement `batch_predict/2`. MLServe detects the absence with `function_exported?/3`
      and maps `predict/2` over the list instead. Declaring a capability the model lacks would
      buy a crash at the first multi-row request.

    * **Sequence pinned to 128** — every input must arrive as exactly 128 tokens. Padding,
      truncation and the matching attention mask are the backend's job, because the caller
      cannot be expected to know the export's shape.
  """

  @behaviour MLServe.Model

  @sequence_length 128
  @vocabulary 32
  @layers 5

  # :exclusive, because an ONNX Runtime session is not guaranteed safe to call concurrently.
  # :once, because the session handle is a NIF resource — one copy of the weights for the pool.
  @impl true
  def capabilities, do: %{concurrency: :exclusive, load: :once}

  @impl true
  def load(config) do
    path = Keyword.fetch!(config, :path)
    {:ok, %{session: Ortex.load(path)}}
  rescue
    error -> {:error, {:ortex_load_failed, Exception.message(error)}}
  end

  # Deliberately no batch_predict/2. See the moduledoc: the export's batch dimension is 1.

  @impl true
  def predict(state, text) when is_binary(text) do
    {ids, mask, token_count} = encode(text)

    outputs =
      Ortex.run(state.session, {
        Nx.tensor([ids], type: :s64),
        Nx.tensor([mask], type: :f32)
      })

    # Eleven tensors come back. A backend's job is to decide what its callers actually want —
    # returning raw runtime tensors would leak the export's shape into every call site and make
    # swapping the model a breaking change.
    logits = outputs |> elem(0) |> Nx.backend_transfer()

    # The last *real* token, not the last padded one.
    last = logits[[0, token_count - 1]]

    {:ok,
     %{
       tokens: token_count,
       truncated?: token_count == @sequence_length and byte_size(text) > @sequence_length,
       top_tokens: top_tokens(last, 3),
       hidden_shape: Nx.shape(logits),
       kv_cache_layers: @layers
     }}
  end

  def predict(_state, other), do: {:error, {:expected_a_string, other}}

  @impl true
  def metadata(_state) do
    %{
      architecture: "gpt-neox",
      sequence_length: @sequence_length,
      vocabulary: @vocabulary,
      layers: @layers,
      batch_dimension: 1,
      trained?: false
    }
  end

  @doc """
  A byte-level stand-in for a tokenizer.

  The real model ships no vocabulary file, so there is nothing honest to tokenize with. Bytes
  modulo the vocabulary size produce valid token ids in range, which is all the shapes require.
  A real backend loads its tokenizer in `load/1` beside the session.
  """
  def encode(text) do
    ids = text |> :binary.bin_to_list() |> Enum.map(&rem(&1, @vocabulary))
    token_count = min(length(ids), @sequence_length)

    padded =
      ids
      |> Enum.take(@sequence_length)
      |> then(&(&1 ++ List.duplicate(0, @sequence_length - length(&1))))

    # 1.0 for real tokens, 0.0 for padding — so attention never reads the pad positions.
    mask = List.duplicate(1.0, token_count) ++ List.duplicate(0.0, @sequence_length - token_count)

    {padded, mask, max(token_count, 1)}
  end

  defp top_tokens(logits, count) do
    order = logits |> Nx.argsort(direction: :desc) |> Nx.to_flat_list() |> Enum.take(count)
    values = Nx.to_flat_list(logits)

    Enum.map(order, fn token -> {token, values |> Enum.at(token) |> Float.round(4)} end)
  end
end

section.("Loading")

{:ok, _} =
  MLServe.load_model(:tiny_gptneox,
    backend: TinyGPTNeoX,
    version: "1.0.0",
    path: "gptneox_Opset18.onnx",
    workers: 2,
    timeout: 5_000,
    # Rejecting bad input in the caller costs one function call. Letting it reach a worker costs
    # a worker slot, and on a real transformer that slot is expensive.
    preprocess: fn
      text when is_binary(text) and byte_size(text) > 0 -> {:ok, text}
      "" -> {:error, :empty_input}
      other -> {:error, {:expected_a_string, other}}
    end,
    cache: [enabled: true, ttl: :timer.minutes(5)]
  )

:ok = MLServe.await_ready(:tiny_gptneox)

{:ok, status} = MLServe.model_status(:tiny_gptneox)

IO.puts("""
    status            #{status.status}
    loaded in         #{status.load_duration_ms}ms
    workers           #{status.workers} (#{status.concurrency})
    native batching   #{status.native_batching}
    metadata          #{inspect(status.metadata)}\
""")

section.("A forward pass")

{:ok, result} = MLServe.predict(:tiny_gptneox, "the quick brown fox")

IO.puts("""
    tokens            #{result.tokens}
    logits shape      #{inspect(result.hidden_shape)}
    kv cache layers   #{result.kv_cache_layers}
    top tokens        #{inspect(result.top_tokens)}

    Real arithmetic over real weights — and meaningless as language, because the weights are
    random. The shapes are the part that is true.\
""")

section.("Why this model cannot batch")

# The contrast with fraud_detection.exs is the lesson. That export has a dynamic batch dimension
# and implements batch_predict/2, so a batch is one Ortex.run. This one cannot, so it does not
# claim it — and MLServe reads the claim rather than assuming.
{:ok, results} =
  MLServe.batch_predict(:tiny_gptneox, ["first input", "second input", "third input"])

IO.puts("""
    batch_predict/3 with 3 inputs → #{length(results)} results
    native_batching                 #{status.native_batching}

    MLServe detects batch_predict/2 with function_exported?/3. It is absent here, so the call
    above became three sequential predict/2 calls — correct, just not faster. Declaring the
    capability anyway would buy a runtime crash at the first two-row tensor.

    This is also why :batching is *not* configured on this model. A window that coalesces
    arrivals into a batch the runtime cannot accept adds latency and nothing else.\
""")

section.("Padding and truncation belong to the backend")

# The export demands exactly 128 tokens. A caller holding a five-character string should not have
# to know that, so the backend owns it.
for text <- ["hi", "a sentence of moderate length", String.duplicate("long ", 60)] do
  {:ok, result} = MLServe.predict(:tiny_gptneox, text)
  {_padded, _mask, count} = TinyGPTNeoX.encode(text)

  label = text |> String.slice(0, 22) |> String.pad_trailing(24)

  IO.puts(
    "    #{label} #{String.pad_leading(to_string(byte_size(text)), 4)} bytes → #{count} tokens, truncated: #{result.truncated?}"
  )
end

IO.puts("""

    Every one ran as a [1, 128] tensor. The attention mask marks which positions are real, so
    padding never contributes to attention.\
""")

section.("Caching an expensive, deterministic call")

# A transformer forward pass is the archetypal cache candidate: costly and — with fixed weights
# and greedy decoding — deterministic. This model is small enough that the win looks modest;
# on a real one the same hit skips tens of milliseconds of GPU time.
defmodule CacheTally do
  @moduledoc "A named handler rather than a capture: :telemetry warns about local functions."

  def start do
    :ets.new(:cache_tally, [:public, :named_table])

    :telemetry.attach_many(
      "cache",
      [[:ml_serve, :cache, :hit], [:ml_serve, :cache, :miss]],
      &__MODULE__.handle_event/4,
      nil
    )
  end

  def handle_event([:ml_serve, :cache, outcome], _measurements, _metadata, _config) do
    :ets.update_counter(:cache_tally, outcome, {2, 1}, {outcome, 0})
    :ok
  end

  def get(key) do
    case :ets.lookup(:cache_tally, key) do
      [{^key, value}] -> value
      [] -> 0
    end
  end
end

CacheTally.start()

time = fn fun ->
  {us, _} = :timer.tc(fun)
  Float.round(us / 1_000, 2)
end

first = time.(fn -> MLServe.predict!(:tiny_gptneox, "a repeated prompt") end)
second = time.(fn -> MLServe.predict!(:tiny_gptneox, "a repeated prompt") end)
third = time.(fn -> MLServe.predict!(:tiny_gptneox, "a repeated prompt") end)

IO.puts("""
    1st call          #{first}ms   (miss — the forward pass ran)
    2nd call          #{second}ms
    3rd call          #{third}ms
    cache hits        #{CacheTally.get(:hit)}

    Caching is only correct because these weights are frozen and decoding is deterministic. A
    model that samples, or reads a clock, must not be cached — which is why MLServe leaves it
    off until you ask.\
""")

section.("Rejecting bad input before it costs a worker")

for input <- [42, %{text: "a map"}, ""] do
  {:error, reason} = MLServe.predict(:tiny_gptneox, input)
  IO.puts("    #{String.pad_trailing(inspect(input), 20)} #{inspect(reason)}")
end

IO.puts("""

    The :preprocess hook runs in the calling process. None of these occupied a worker, and on a
    model that takes 40ms a request that was never going to succeed is 40ms of pool time saved.\
""")

section.("Two models, two shapes, one interface")

IO.puts("""
                          fraud_detection.onnx      gptneox_Opset18.onnx
    batch dimension       dynamic                   pinned to 1
    inputs                1                         2, different dtypes
    outputs               1                         11 (logits + KV cache)
    batch_predict/2       implemented               deliberately absent
    :batching             max_size: 32              would be actively harmful
    concurrency           :exclusive                :exclusive
    load                  :once                     :once

    The callers of both write MLServe.predict(name, input). Everything above is a property of the
    export, declared once by the backend, and never leaked into a call site.\
""")

MLServe.unload_model(:tiny_gptneox)
