import Config

# The point of MLServe.Backend.Static: the test suite exercises the real router, the real error
# mapping, real caching and real telemetry, with no model file, no ML runtime and no GPU. Only
# the thing being stubbed is stubbed.
config :ml_serve,
  start_mode: :async,
  models: [
    fraud_detection: [
      backend: MLServe.Backend.Static,
      version: "1.0.0",
      config: [result: %{prediction: "fraud", probability: 0.94}]
    ],
    sentiment: [
      backend: MLServe.Backend.Static,
      version: "1.0.0",
      config: [result: %{label: "positive", score: 0.88}]
    ],
    # Every service needs a test for the unhappy path too.
    always_failing: [
      backend: MLServe.Backend.Static,
      version: "1.0.0",
      config: [error: :upstream_unavailable]
    ]
  ]

config :inference_service, port: 4002

config :logger, level: :warning
