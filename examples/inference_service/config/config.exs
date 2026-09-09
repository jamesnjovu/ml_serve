import Config

# Models declared here are loaded during application boot by MLServe's own supervision tree.
# This is the same validation and the same code path as MLServe.load_model/2 at runtime — the
# only difference is who calls it.
config :ml_serve,
  # :async means a slow model load never blocks boot. The model is registered immediately and
  # reports :loading until it is ready, which is exactly what GET /health reads.
  start_mode: :async,
  models: [
    # An :exclusive model: a worker pool, because the backend is assumed unsafe to call
    # concurrently. Results are cached because this model is deterministic.
    fraud_detection: [
      backend: InferenceService.Models.Fraud,
      version: "1.0.0",
      workers: 4,
      timeout: 2_000,
      max_concurrency: 200,
      cache: [enabled: true, ttl: :timer.minutes(1)],
      config: [threshold: 0.7]
    ],

    # A :shared model: inference runs in the caller with no worker pool at all, and dynamic
    # batching coalesces concurrent single requests into one backend call.
    sentiment: [
      backend: InferenceService.Models.Sentiment,
      version: "1.0.0",
      workers: 4,
      batching: [max_size: 16, timeout: 10],
      config: []
    ]
  ]

config :inference_service, port: 4000

config :logger, :console, format: "$time $metadata[$level] $message\n"

import_config "#{config_env()}.exs"
