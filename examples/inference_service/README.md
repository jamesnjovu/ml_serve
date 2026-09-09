# Inference Service

A JSON inference API built on [MLServe](https://github.com/jamesnjovu/ml_serve), using
[Bandit](https://hex.pm/packages/bandit) and [Plug](https://hex.pm/packages/plug).

It is deliberately small, but nothing here is a toy: models are declared in configuration and
loaded by MLServe's own supervision tree, the two models exercise both execution strategies, and
the error mapping is the one you would actually ship.

## Running it

```bash
mix deps.get
mix run --no-halt
```

Then:

```bash
curl localhost:4000/health

curl -X POST localhost:4000/predict/fraud_detection \
  -H 'content-type: application/json' \
  -d '{"amount": 1500.50, "failed_transactions_24h": 2, "transaction_count_24h": 8}'
#=> {"result":{"prediction":"fraud","probability":0.8902,"threshold":0.7}}

curl -X POST localhost:4000/predict/sentiment/batch \
  -H 'content-type: application/json' \
  -d '{"inputs": [{"text": "this is great"}, {"text": "awful and broken"}]}'
#=> {"count":2,"results":[{"label":"positive",...},{"label":"negative",...}]}

curl localhost:4000/models | jq
```

## Endpoints

| Method | Path | |
| ------ | ---- | --- |
| `GET` | `/health` | Readiness probe. 200 once every declared model has loaded, 503 while any is still loading. |
| `GET` | `/models` | Every model with its live status, request and error counters. |
| `GET` | `/models/:name` | One model in detail. |
| `POST` | `/predict/:name` | One prediction. The JSON body is the input. |
| `POST` | `/predict/:name/batch` | Many predictions, from `{"inputs": [...]}`. |

Query parameters on both predict routes:

| Parameter | |
| --------- | --- |
| `version` | Pin an exact model version, bypassing default routing and any canary. |
| `cache` | `true` or `false`, overriding the model's configured cache setting. |
| `timeout` | Per-request deadline in milliseconds. |

## What each piece demonstrates

**`config/config.exs`** — models declared in configuration rather than loaded by hand. MLServe
boots them itself; the application supervision tree in `lib/inference_service/application.ex`
starts only the HTTP endpoint. `start_mode: :async` means a slow model load never blocks boot,
which is why `/health` exists and why it flips to 200 on its own.

**`Models.Fraud`** — `concurrency: :exclusive` with a pool of four workers and `load: :once`, so
one shared state serves the whole pool. The shape for an ONNX session or a port to a Python
process. Results are cached because the model is deterministic.

**`Models.Sentiment`** — `concurrency: :shared`, so inference runs in the Bandit request process
itself, with no worker pool and no message copies. It implements the optional `batch_predict/2`,
so the `batching: [max_size: 16, timeout: 10]` window turns sixteen concurrent HTTP requests into
one backend call.

**`ErrorMapping`** — the part worth stealing. MLServe's error taxonomy mapped onto HTTP status
codes, driven by `MLServe.Error.retryable?/1`:

| MLServe reason | HTTP | |
| -------------- | ---- | --- |
| `:model_not_found` | 404 | A client bug. |
| `:model_not_ready` | 503 | Loading, draining or failed. `Retry-After: 1`. |
| `:timeout` | 504 | The deadline passed. |
| `:overloaded` | 429 | Backpressure, not failure. `Retry-After: 1`. |
| `{:invalid_input, _}` | 422 | The request body was wrong. |
| `{:batch_too_large, max}` | 413 | Over `:max_batch_size`. |
| `{:backend_error, _}` | 500 | Ours to fix. The stacktrace goes to the log, never to the client. |

A 500 where a 429 belonged turns backpressure into an outage, because client retry policies treat
them differently. That is the whole reason this module is separate and separately tested.

**`Telemetry`** — logs failed and slow predictions, splitting queue time from inference time so
"the model got slower" and "the pool is too small" stay distinguishable. A real deployment would
use `MLServe.Telemetry.Metrics.metrics/0` and a reporter instead.

## Tests

```bash
mix test
```

`config/test.exs` points every model at `MLServe.Backend.Static`, so the suite exercises the real
router, the real error mapping, real caching and real telemetry with no model file, no ML runtime
and no mocking library. Only the model is stubbed.
