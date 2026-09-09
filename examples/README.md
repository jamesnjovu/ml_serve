# MLServe Examples

Three ways in, depending on how you like to learn. Everything here runs against the library in
this repository — no published package required.

| | |
| --- | --- |
| **[`scripts/`](scripts)** | Seven self-contained scripts. One command each, no setup. |
| **[`notebooks/`](notebooks)** | Three Livebook notebooks. Same ground, interactive. |
| **[`inference_service/`](inference_service)** | A real HTTP service — Bandit, Plug, JSON. |
| **[`onnx/`](onnx)** | Three `.onnx` models through ONNX Runtime, including a real HuggingFace one. |

None of this ships in the Hex package; it lives here on GitHub.

## Scripts

Each is a single file that installs its own dependencies and runs in a few seconds:

```bash
elixir examples/scripts/01_quick_start.exs
```

| | |
| --- | --- |
| [`01_quick_start.exs`](scripts/01_quick_start.exs) | Define a backend, load it, predict, read status, unload. |
| [`02_concurrency.exs`](scripts/02_concurrency.exs) | `:shared` vs `:exclusive`, worker pools, and shedding load instead of queueing it. |
| [`03_batching.exs`](scripts/03_batching.exs) | 200 concurrent callers, 5 backend calls, ~10× faster. |
| [`04_caching.exs`](scripts/04_caching.exs) | Why it is off by default, cheap keys, TTL, and what is never cached. |
| [`05_versioning.exs`](scripts/05_versioning.exs) | Two versions side by side, a 20% canary, promotion, draining. |
| [`06_telemetry.exs`](scripts/06_telemetry.exs) | Latency percentiles, queue time vs inference time, LiveDashboard metrics. |
| [`07_error_handling.exs`](scripts/07_error_handling.exs) | Every error MLServe can return, triggered for real. |

They print rather than assert, so the output *is* the documentation:

```
── Spreading load across the pool ────────────────────────────────
  4 distinct workers served 200 requests:
    #PID<0.214.0>  ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 50
    #PID<0.215.0>  ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 50
    #PID<0.216.0>  ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 50
    #PID<0.217.0>  ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 50
```

## Notebooks

Open in [Livebook](https://livebook.dev):

| | |
| --- | --- |
| [`quick_start.livemd`](notebooks/quick_start.livemd) | The basics, about five minutes. |
| [`versioning_and_canary.livemd`](notebooks/versioning_and_canary.livemd) | Side-by-side versions, canary splits, a candidate that cannot load. |
| [`batching_and_caching.livemd`](notebooks/batching_and_caching.livemd) | Two different ways to not do work, and when each applies. |

## The HTTP service

A JSON inference API on Bandit and Plug, with models declared in configuration, both execution
strategies in use, and a full MLServe-error-to-HTTP-status mapping:

```bash
cd examples/inference_service
mix deps.get
mix run --no-halt

curl -X POST localhost:4000/predict/fraud_detection \
  -H 'content-type: application/json' \
  -d '{"amount": 1500.50, "failed_transactions_24h": 2, "transaction_count_24h": 8}'
```

It has [its own README](inference_service/README.md) and a test suite that stubs the model with
`MLServe.Backend.Static` while exercising the real router, error mapping, caching and telemetry.

## Real ONNX models

Everything above uses arithmetic in place of a model, which keeps the mechanics visible.
[`onnx/`](onnx) is the exception: actual `.onnx` files loaded through ONNX Runtime via
[Ortex](https://hex.pm/packages/ortex).

```bash
elixir examples/onnx/fraud_detection.exs      # a convenient export
elixir examples/onnx/transformer.exs          # an awkward one
elixir examples/onnx/sentence_embeddings.exs  # a real, trained one
```

They are a set on purpose. The first is shaped the way tutorials assume — a dynamic batch
dimension, one input, one output — and covers what only matters once a model is a file: `:path`
containment against `:model_root`, `:checksum` enforcement, one session shared across the pool
via `load: :once`, and a batch that becomes a single `Ortex.run/2`.

The second is a PyTorch-exported GPT-NeoX with the constraints real exports actually have: a
batch dimension pinned to `1`, a sequence length pinned to `128`, two inputs of different dtypes
and eleven outputs. It shows the case tutorials skip — a backend that must *not* implement
`batch_predict/2`, and must own padding, truncation and the attention mask itself.

The third is [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
straight off the HuggingFace Hub, and it is the one you would actually deploy. Trained weights,
so the output means something — a paraphrase scores 0.973 and unrelated sentences score zero —
ending in real semantic search over a corpus. It brings a WordPiece tokenizer inside the backend
(verified token-for-token against HuggingFace's own), mean pooling with an attention mask, and
the production artifact story: the 90 MB model is fetched at first run and checksum-verified
rather than committed.

The first run of any of them compiles Ortex's Rust NIF, which takes a few minutes; later runs are
instant. The two small model files are committed, and the smallest ships with the
[script that generated it](onnx/generate_model.py).

## Against the published package

Every example depends on the library by path so it runs from a clone. To point one at Hex
instead, swap the dependency:

```elixir
# {:ml_serve, path: Path.expand("../..", __DIR__)}
{:ml_serve, "~> 0.1.0"}
```

## Guides

The examples show; the [guides](../guides) explain. Start with
[Getting Started](../guides/getting-started.md) and
[Creating a Model Backend](../guides/creating-a-backend.md), which has complete Nx, Bumblebee,
ONNX and Python implementations.
