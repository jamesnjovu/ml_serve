# Serving ONNX models

Everywhere else in `examples/` the "model" is arithmetic — enough to show MLServe's mechanics
without dragging in an ML runtime. These two are real: `.onnx` files on disk, loaded through ONNX
Runtime via [Ortex](https://hex.pm/packages/ortex), producing genuine inference.

```bash
elixir examples/onnx/fraud_detection.exs      # a convenient export
elixir examples/onnx/transformer.exs          # an awkward one
elixir examples/onnx/sentence_embeddings.exs  # a real, trained one
```

The first run of any of them compiles Ortex's Rust NIF and downloads ONNX Runtime — a few
minutes. Later runs start immediately.

They are a set on purpose. Almost every ONNX tutorial uses a model shaped like the first and
leaves you unprepared for the second, which is what real PyTorch exports often look like. The
third is the one you would actually deploy.

| | `fraud_detection.onnx` | `gptneox_Opset18.onnx` | `all-MiniLM-L6-v2` |
| --- | --- | --- | --- |
| Trained? | hand-written weights | **no — random** | **yes** |
| Batch dimension | dynamic | **pinned to 1** | dynamic |
| Sequence | n/a | **pinned to 128** | dynamic, capped at 256 |
| Inputs | 1 | 2 | 3 |
| Outputs | 1 | 11 (logits + KV cache) | 1 |
| `batch_predict/2` | implemented | **deliberately absent** | implemented |
| `:batching` | `max_size: 32` | would be actively harmful | `max_size: 16` |
| Preprocessing | feature scaling | padding + mask | **WordPiece tokenizer** |
| Size | 323 bytes | 1.8 MB | 90 MB, **not committed** |

Callers of all three write `MLServe.predict(name, input)`. Every difference above is a property
of the export, declared once by the backend, and never leaked into a call site.

---

## `fraud_detection.exs` — a model as a file

A logistic regression (`MatMul → Add → Sigmoid`) over three normalised features. It exists to
show the parts of MLServe that only start to matter once a model is a **file**:

**`:path` is validated before your backend sees it.** `MLServe.Security` resolves the path,
confirms it sits inside `:model_root`, follows symlinks and re-checks containment, and confirms a
readable regular file within `:max_model_bytes`. A model artifact is data, never code, and the
path naming it is attacker-influenced in more deployments than people expect:

```
fraud_detection.onnx   accepted — fraud_detection.onnx
../../etc/passwd       rejected — outside_root
/etc/hosts             rejected — outside_root
missing.onnx           rejected — enoent
```

**`:checksum` is enforced, not decorative.** The file is hashed before `load/1` is called. The
example loads the same file with a deliberately wrong digest and shows it retrying with backoff,
settling into `:failed` with `{:checksum_mismatch, actual}` preserved — and `load/1` never having
run, because the artifact is rejected before the backend sees it.

**`load: :once` means one session for the whole pool.** A session handle is a NIF resource: cheap
to copy, backed by memory the runtime owns. Four workers share one session and therefore one copy
of the weights. `load: :per_worker` would multiply that by the pool size, which on a GPU is the
difference between fitting and not. The example proves it by printing from `load/1` and showing
the line appear once against `workers: 4`.

**`batch_predict/2` is one `Ortex.run/2`.** The batch becomes a single `[N, 3]` tensor, so the
fixed cost of crossing into the runtime is paid once instead of per row — which is exactly what
the `batching: [max_size: 32, timeout: 10]` window exists to feed.

```
── Real inference ────────────────────────────────────────────────
    small purchase     0.0832  legitimate
    unusual            0.7027  fraud
    card testing       1.0000  fraud
```

### Where the model comes from

`fraud_detection.onnx` is 323 bytes and committed, so the example runs with no build step.
[`generate_model.py`](generate_model.py) sits beside it so the artifact is reproducible rather
than magic:

```bash
pip install onnx
python examples/onnx/generate_model.py
```

A real project exports from scikit-learn or PyTorch; the graph is hand-written here only to keep
the example free of a training dependency.

> **One exporter gotcha.** Recent `onnx` releases default to an IR version newer than the ONNX
> Runtime bundled with Ortex accepts, and it surfaces as an opaque `Unsupported model IR version`
> at load rather than anything pointing back at the exporter. `generate_model.py` pins
> `model.ir_version = 8`.

---

## `transformer.exs` — constraints the export imposes on you

`gptneox_Opset18.onnx` is a GPT-NeoX exported from PyTorch 2.1.0 at opset 18: 658 nodes, five
layers, four heads, a 32-token vocabulary.

> **The weights are random.** This is a shape-and-opset conformance model — correct architecture,
> untrained weights. Every logit it produces is real arithmetic over real weights, and every one
> of them is meaningless as language. It is here to exercise MLServe against a model with a
> transformer's *shape*.

```
input_ids        int64    [1, 128]
attention_mask   float32  [1, 128]
logits           float32  [1, 128, 32]
key/value × 5    float32  [1, 4, 128, 8]
```

Both leading dimensions are literal, not symbolic, and that drives everything:

**A backend must not claim a capability the model lacks.** The runtime physically cannot accept
two rows, so this backend does not implement `batch_predict/2`. MLServe detects the absence with
`function_exported?/3` and maps `predict/2` over the list instead — correct, just not faster.
`MLServe.model_status/2` reports `native_batching: false`. Implementing it anyway would buy a
runtime crash at the first two-row tensor.

**Which is also why `:batching` is not configured.** A window that coalesces arrivals into a
batch the runtime cannot accept adds latency and nothing else.

**Padding, truncation and the attention mask are the backend's job.** Every input must arrive as
exactly 128 tokens; a caller holding a five-character string should not have to know that:

```
hi                          2 bytes → 2 tokens, truncated: false
a sentence of moderate     29 bytes → 29 tokens, truncated: false
long long long long lo    300 bytes → 128 tokens, truncated: true
```

**Eleven output tensors, and the backend decides what callers see.** Returning raw runtime
tensors would leak the export's shape into every call site and make swapping the model a breaking
change.

**A `:preprocess` hook rejects bad input in the calling process**, before a worker slot is spent.
On a real transformer, a request that was never going to succeed is tens of milliseconds of pool
time saved.

**Caching, with the caveat stated.** A forward pass is costly and — with frozen weights and
greedy decoding — deterministic, which makes it the archetypal cache candidate. A model that
samples or reads a clock must not be cached, which is why MLServe leaves caching off until asked.

### The tokenizer

There is none: the model ships no vocabulary file, so there is nothing honest to tokenize with.
The example uses bytes modulo the vocabulary size, which produces valid in-range token ids and is
all the shapes require. A real backend loads its tokenizer in `load/1` beside the session.

---

## `sentence_embeddings.exs` — a model you would actually deploy

[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
the most-downloaded sentence embedding model on the Hub, served through ONNX Runtime. This one is
*trained*, so the output means something:

```
           [0]     [1]     [2]     [3]     [4]
[0]    1.000   0.973   0.537  -0.021  -0.044   The cat sits on the mat
[1]    0.973   1.000   0.545  -0.027  -0.020   A cat is sitting on a mat
[2]    0.537   0.545   1.000   0.023  -0.046   A feline rests upon a rug
[3]   -0.021  -0.027   0.023   1.000   0.028   Quantum physics is complicated
[4]   -0.044  -0.020  -0.046   0.028   1.000   Machine learning models need training data
```

A paraphrase scores 0.973; the same idea in entirely different words lands at 0.54; unrelated
sentences sit at zero. The example then does the thing embeddings are for — ranking a corpus
against a query by cosine similarity:

```
"How do I keep a crashed process alive?"
  0.432  Supervision trees restart failed processes
  0.294  GenServer is the standard stateful process abstraction

"What temperature for bread?"
  0.466  Preheat the oven to 220 degrees before baking
  0.251  Sourdough needs a starter and long fermentation
```

**The 90 MB model is not committed.** It is fetched from the Hub on first run into `.models/`
(gitignored) and verified against a pinned SHA-256 — which is the artifact story from
[`guides/production-deployment.md`](../../guides/production-deployment.md) rather than a shortcut
around it. MLServe re-verifies the model at load via `:checksum`; the example verifies
`vocab.txt` itself, because a half-written tokenizer is a far more confusing failure than a loud
one.

**Tokenization lives in the backend.** BERT WordPiece is implemented in ~70 lines from the
model's own `vocab.txt` — lowercasing, accent stripping, punctuation splitting and greedy
longest-match subwording. It was verified token-for-token against HuggingFace's `tokenizers`
across accents (`naïve café résumé`), contractions (`Don't panic, it's fine!`), subword splits
(`supercalifragilisticexpialidocious`), numerals (`COVID-19 vaccines (2021) cost $19.99`) and
casing. Real projects reach for `Bumblebee` or the `tokenizers` NIF; doing it by hand here keeps
the example free of a second Rust dependency and makes the point that preprocessing belongs
*inside* a backend, where no caller can get it subtly wrong.

**Pooling is a decision the backend owns.** The graph emits per-token hidden states
`[batch, seq, 384]`. A *sentence* embedding is mean pooling over non-padding tokens followed by
L2 normalisation — that is `1_Pooling/config.json` in the Hub repo, not something the ONNX graph
does. Callers get a unit vector they can dot together and never learn that pooling was a choice.

**Both dimensions are dynamic**, which is what `transformer.exs` is not. So `batch_predict/2` is
implemented and means it, padding to the longest sequence in *this* batch rather than a fixed
export width. 64 concurrent single-sentence callers become 3 forward passes:

```
64 concurrent callers in 125.5ms
backend invocations      3
rows                     64
```

---

## Related

* [`guides/creating-a-backend.md`](../../guides/creating-a-backend.md) — the ONNX section, plus
  Nx, Bumblebee and Python-over-a-port backends.
* [`guides/production-deployment.md`](../../guides/production-deployment.md) — worker counts per
  runtime, and shipping a new artifact without a deploy.
* [`examples/scripts/03_batching.exs`](../scripts/03_batching.exs) — what a batching window does
  underneath, on a model that can actually use one.
