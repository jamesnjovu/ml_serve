# Semantic search with a real HuggingFace model.
#
#     elixir examples/onnx/sentence_embeddings.exs
#
# sentence-transformers/all-MiniLM-L6-v2, the most-downloaded sentence embedding model on the
# Hub, served through ONNX Runtime. Unlike the other two ONNX examples this one is *trained*, so
# the numbers at the end mean something: sentences that are actually about the same thing score
# high, and sentences that are not score near zero.
#
# The 90 MB model is not committed. It is fetched from the Hub on first run into .models/ and
# verified against a pinned SHA-256 — which is the artifact story from
# guides/production-deployment.md, not a shortcut around it.
#
# The first run also compiles Ortex's Rust NIF and downloads ONNX Runtime, which takes a few
# minutes. Later runs start immediately.

Mix.install([
  # Running from a clone of the repo. Against the published package this is:
  #   {:ml_serve, "~> 0.1.0"}
  {:ml_serve, path: Path.expand("../..", __DIR__)},
  {:ortex, "~> 0.1.10"},
  {:nx, "~> 0.7"},
  {:req, "~> 0.5"}
])

section = fn title ->
  IO.puts("\n── " <> title <> " " <> String.duplicate("─", max(0, 62 - String.length(title))))
end

model_root = Path.join(__DIR__, ".models")
Application.put_env(:ml_serve, :model_root, model_root)

defmodule Artifact do
  @moduledoc """
  Fetches the model from the Hub on first run.

  This is the pattern in `guides/production-deployment.md`: large artifacts are not in the repo
  and not in the image — they are pulled at boot from wherever they live, then handed to
  `MLServe.load_model/2` with a `:checksum`, so a truncated download or a swapped file fails
  loudly at load rather than quietly at inference.
  """

  @base "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

  @files %{
    "model.onnx" =>
      {"#{@base}/onnx/model.onnx",
       "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"},
    "vocab.txt" =>
      {"#{@base}/vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"}
  }

  def checksum(name), do: @files |> Map.fetch!(name) |> elem(1)

  def ensure_downloaded!(root) do
    File.mkdir_p!(root)

    Enum.each(@files, fn {name, {url, expected}} ->
      path = Path.join(root, name)

      if fresh?(path, expected) do
        IO.puts("    #{String.pad_trailing(name, 12)} cached")
      else
        IO.puts("    #{String.pad_trailing(name, 12)} downloading…")
        Req.get!(url, into: File.stream!(path), redirect: true, redirect_log_level: false)

        # Verify immediately. MLServe re-verifies the model at load, but vocab.txt never reaches
        # MLServe, and a half-written tokenizer is a far more confusing failure than a loud one.
        {:ok, actual} = MLServe.Security.digest(path, :sha256)

        if actual != expected do
          File.rm(path)
          raise "checksum mismatch for #{name}: expected #{expected}, got #{actual}"
        end

        IO.puts(
          "    #{String.pad_trailing(name, 12)} #{File.stat!(path).size} bytes, checksum ok"
        )
      end
    end)
  end

  defp fresh?(path, expected) do
    File.exists?(path) and match?({:ok, ^expected}, MLServe.Security.digest(path, :sha256))
  end
end

defmodule WordPiece do
  @moduledoc """
  BERT WordPiece tokenization, from the model's own `vocab.txt`.

  Verified token-for-token against HuggingFace's `tokenizers` across accents, contractions,
  punctuation, subword splits, numerals and casing. Real projects reach for `Bumblebee` or the
  `tokenizers` NIF; this is here so the example has no Rust dependency beyond Ortex, and because
  a tokenizer is exactly the kind of preprocessing that belongs *inside* a backend.
  """

  @unk "[UNK]"
  @cls "[CLS]"
  @sep "[SEP]"

  def load_vocab(path) do
    path
    |> File.stream!()
    |> Stream.map(&String.trim_trailing(&1, "\n"))
    |> Stream.with_index()
    |> Map.new()
  end

  @doc "Returns `[CLS] … [SEP]` token ids, truncated to `max_length`."
  def encode(vocab, text, max_length) do
    ids =
      text
      |> basic_tokenize()
      |> Enum.flat_map(&wordpiece(vocab, &1))
      |> Enum.take(max_length - 2)
      |> Enum.map(&Map.fetch!(vocab, &1))

    [Map.fetch!(vocab, @cls)] ++ ids ++ [Map.fetch!(vocab, @sep)]
  end

  defp basic_tokenize(text) do
    text
    |> String.downcase()
    |> strip_accents()
    |> String.split(~r/\s+/, trim: true)
    |> Enum.flat_map(&split_punctuation/1)
  end

  # This model is uncased and strips accents: "café" and "cafe" must produce identical ids.
  defp strip_accents(text) do
    text
    |> :unicode.characters_to_nfd_binary()
    |> String.to_charlist()
    |> Enum.reject(&(&1 in 0x0300..0x036F))
    |> List.to_string()
  end

  defp split_punctuation(word) do
    word
    |> String.to_charlist()
    |> Enum.chunk_by(&punctuation?/1)
    |> Enum.flat_map(fn chunk ->
      if punctuation?(hd(chunk)),
        do: Enum.map(chunk, &List.to_string([&1])),
        else: [List.to_string(chunk)]
    end)
  end

  defp punctuation?(c), do: c in 33..47 or c in 58..64 or c in 91..96 or c in 123..126

  # Greedy longest-match-first; continuations carry the ## prefix.
  defp wordpiece(vocab, word, max_chars \\ 100) do
    graphemes = String.graphemes(word)

    if length(graphemes) > max_chars do
      [@unk]
    else
      case walk(vocab, graphemes, 0, length(graphemes), []) do
        :error -> [@unk]
        tokens -> tokens
      end
    end
  end

  defp walk(_vocab, _graphemes, start, len, acc) when start >= len, do: Enum.reverse(acc)

  defp walk(vocab, graphemes, start, len, acc) do
    case longest(vocab, graphemes, start, len, start != 0) do
      nil -> :error
      {token, stop} -> walk(vocab, graphemes, stop, len, [token | acc])
    end
  end

  defp longest(vocab, graphemes, start, stop, continuation?) when stop > start do
    piece = graphemes |> Enum.slice(start, stop - start) |> Enum.join()
    candidate = if continuation?, do: "##" <> piece, else: piece

    if Map.has_key?(vocab, candidate),
      do: {candidate, stop},
      else: longest(vocab, graphemes, start, stop - 1, continuation?)
  end

  defp longest(_vocab, _graphemes, _start, _stop, _continuation?), do: nil
end

defmodule MiniLM do
  @moduledoc """
  all-MiniLM-L6-v2 as an `MLServe.Model`.

  ## What the export allows

      input_ids        int64    [batch_size, sequence_length]
      attention_mask   int64    [batch_size, sequence_length]
      token_type_ids   int64    [batch_size, sequence_length]
      last_hidden_state float32 [batch_size, sequence_length, 384]

  Both dimensions are *symbolic*, which is the opposite of `transformer.exs`. So this backend
  implements `batch_predict/2` and means it: N sentences become one `[N, S, 384]` forward pass,
  padded to the longest sequence in that batch rather than to a fixed export length.

  ## Where the model ends and the backend begins

  The graph emits per-token hidden states. A *sentence* embedding is mean pooling over the
  non-padding tokens followed by L2 normalisation — that is the `1_Pooling/config.json` in the
  Hub repo, not something the ONNX graph does. Doing it here means callers get a unit vector they
  can dot together, and never have to know that pooling was a choice.
  """

  @behaviour MLServe.Model

  # sentence_bert_config.json. Longer inputs are truncated rather than rejected, matching
  # sentence-transformers' own behaviour.
  @max_sequence 256

  @impl true
  def capabilities, do: %{concurrency: :exclusive, load: :once}

  @impl true
  def load(config) do
    path = Keyword.fetch!(config, :path)
    vocab_path = Keyword.fetch!(config, :vocab)

    # Both the session and the vocabulary are built once and shared by the whole pool. The vocab
    # is a 30,522-entry map; loading it per worker would be pure waste.
    {:ok, %{session: Ortex.load(path), vocab: WordPiece.load_vocab(vocab_path)}}
  rescue
    error -> {:error, {:load_failed, Exception.message(error)}}
  end

  @impl true
  def predict(state, text) when is_binary(text) do
    with {:ok, [embedding]} <- batch_predict(state, [text]), do: {:ok, embedding}
  end

  def predict(_state, other), do: {:error, {:expected_a_string, other}}

  @impl true
  def batch_predict(state, texts) do
    if Enum.all?(texts, &is_binary/1) do
      {:ok, embed(state, texts)}
    else
      {:error, {:expected_strings, texts}}
    end
  end

  @impl true
  def metadata(_state) do
    %{
      model: "sentence-transformers/all-MiniLM-L6-v2",
      dimensions: 384,
      max_sequence: @max_sequence,
      pooling: "mean over non-padding tokens, then L2 normalised"
    }
  end

  defp embed(state, texts) do
    encoded = Enum.map(texts, &WordPiece.encode(state.vocab, &1, @max_sequence))

    # Pad to the longest sequence in *this* batch, not to @max_sequence. The export's sequence
    # dimension is dynamic, so padding to a fixed length would be compute spent on [PAD].
    width = encoded |> Enum.map(&length/1) |> Enum.max()

    ids = Enum.map(encoded, &(&1 ++ List.duplicate(0, width - length(&1))))

    mask =
      Enum.map(encoded, &(List.duplicate(1, length(&1)) ++ List.duplicate(0, width - length(&1))))

    types = Enum.map(ids, fn _ -> List.duplicate(0, width) end)

    {hidden} =
      Ortex.run(state.session, {
        Nx.tensor(ids, type: :s64),
        Nx.tensor(mask, type: :s64),
        Nx.tensor(types, type: :s64)
      })

    hidden
    |> Nx.backend_transfer()
    |> mean_pool(Nx.tensor(mask, type: :f32))
    |> normalise()
    |> Nx.to_list()
  end

  # Masked mean: sum the real tokens, divide by how many there were. Averaging over the padded
  # width instead would shrink every short sentence's embedding toward zero.
  defp mean_pool(hidden, mask) do
    expanded = Nx.new_axis(mask, -1)
    summed = hidden |> Nx.multiply(expanded) |> Nx.sum(axes: [1])
    counts = mask |> Nx.sum(axes: [1], keep_axes: true) |> Nx.max(1.0e-9)

    Nx.divide(summed, counts)
  end

  # Unit vectors, so cosine similarity is a plain dot product.
  defp normalise(embeddings) do
    norms = embeddings |> Nx.LinAlg.norm(axes: [1], keep_axes: true) |> Nx.max(1.0e-12)
    Nx.divide(embeddings, norms)
  end
end

section.("Fetching the model")

Artifact.ensure_downloaded!(model_root)

section.("Loading")

{:ok, _} =
  MLServe.load_model(:embeddings,
    backend: MiniLM,
    version: "1.0.0",
    path: "model.onnx",
    # MLServe hashes the 90 MB artifact before the backend is handed the path.
    checksum: {:sha256, Artifact.checksum("model.onnx")},
    workers: 2,
    timeout: 30_000,
    # The export batches, so a window is worth having: concurrent single-sentence callers become
    # one forward pass.
    batching: [max_size: 16, timeout: 20],
    cache: [enabled: true, ttl: :timer.minutes(10)],
    config: [vocab: Path.join(model_root, "vocab.txt")]
  )

:ok = MLServe.await_ready(:embeddings, 60_000)

{:ok, status} = MLServe.model_status(:embeddings)

IO.puts("""
    status            #{status.status}
    loaded in         #{status.load_duration_ms}ms
    workers           #{status.workers} (#{status.concurrency})
    native batching   #{status.native_batching}
    metadata          #{inspect(status.metadata)}\
""")

section.("Does it actually understand anything?")

sentences = [
  "The cat sits on the mat",
  "A cat is sitting on a mat",
  "A feline rests upon a rug",
  "Quantum physics is complicated",
  "Machine learning models need training data"
]

{:ok, embeddings} = MLServe.batch_predict(:embeddings, sentences)

similarity = fn a, b ->
  Enum.zip(a, b) |> Enum.map(fn {x, y} -> x * y end) |> Enum.sum()
end

IO.puts(
  "    #{String.duplicate(" ", 6)}" <> Enum.map_join(0..4, "", &String.pad_leading("[#{&1}]", 8))
)

for {row, i} <- Enum.with_index(embeddings) do
  cells =
    Enum.map_join(embeddings, "", fn other ->
      String.pad_leading(:erlang.float_to_binary(similarity.(row, other), decimals: 3), 8)
    end)

  IO.puts("    [#{i}] #{cells}   #{Enum.at(sentences, i)}")
end

IO.puts("""

    [0]·[1] is a paraphrase and scores highest. [2] says the same thing with entirely different
    words and lands in the middle — this model matches meaning, but it is not magic. [3] and [4]
    share no subject with anything and sit at zero. That spread is the model working.\
""")

section.("Semantic search")

# The point of an embedding model. Embed a corpus once, embed the query, rank by dot product —
# which is cosine similarity, because everything is already normalised.
corpus = [
  "Elixir runs on the Erlang virtual machine",
  "Supervision trees restart failed processes",
  "Sourdough needs a starter and long fermentation",
  "GenServer is the standard stateful process abstraction",
  "Preheat the oven to 220 degrees before baking",
  "Pattern matching destructures data in function heads"
]

{:ok, corpus_vectors} = MLServe.batch_predict(:embeddings, corpus)

for query <- ["How do I keep a crashed process alive?", "What temperature for bread?"] do
  {:ok, query_vector} = MLServe.predict(:embeddings, query)

  IO.puts("\n    #{inspect(query)}")

  corpus
  |> Enum.zip(corpus_vectors)
  |> Enum.map(fn {text, vector} -> {similarity.(query_vector, vector), text} end)
  |> Enum.sort(:desc)
  |> Enum.take(3)
  |> Enum.each(fn {score, text} ->
    IO.puts("      #{:erlang.float_to_binary(score, decimals: 3)}  #{text}")
  end)
end

section.("Batching a burst of independent callers")

# 64 processes, each holding one sentence. Exactly the shape dynamic batching exists for.
:ets.new(:flushes, [:public, :named_table])

defmodule Flushes do
  def attach do
    :telemetry.attach("flush", [:ml_serve, :batch, :flush], &__MODULE__.handle/4, nil)
  end

  def handle(_event, %{size: size}, _metadata, _config) do
    :ets.update_counter(:flushes, :count, {2, 1}, {:count, 0})
    :ets.update_counter(:flushes, :rows, {2, size}, {:rows, 0})
    :ok
  end

  def get(key) do
    case :ets.lookup(:flushes, key) do
      [{^key, value}] -> value
      [] -> 0
    end
  end
end

Flushes.attach()

inputs = Enum.map(1..64, &"sentence number #{&1} about an entirely unrelated subject")

{us, _} =
  :timer.tc(fn ->
    inputs
    |> Task.async_stream(&MLServe.predict!(:embeddings, &1), max_concurrency: 64, timeout: 60_000)
    |> Stream.run()
  end)

IO.puts("""
    64 concurrent callers in #{Float.round(us / 1_000, 1)}ms
    backend invocations      #{Flushes.get(:count)}
    rows                     #{Flushes.get(:rows)}

    Each caller asked for one embedding and MLServe handed the runtime a handful of batches. The
    contrast with transformer.exs is the whole reason both examples exist: there the export's
    batch dimension is pinned to 1 and a window like this would be worse than useless.\
""")

section.("Caching identical text")

time = fn fun ->
  {us, _} = :timer.tc(fun)
  Float.round(us / 1_000, 2)
end

cold = time.(fn -> MLServe.predict!(:embeddings, "a sentence worth remembering") end)
warm = time.(fn -> MLServe.predict!(:embeddings, "a sentence worth remembering") end)

{:ok, final} = MLServe.model_status(:embeddings)

IO.puts("""
    first call        #{cold}ms
    second call       #{warm}ms

    Embeddings are the ideal cache entry: expensive, and deterministic given frozen weights.
    Requests served in this run: #{final.requests}, errors: #{final.errors}.\
""")

MLServe.unload_model(:embeddings)
