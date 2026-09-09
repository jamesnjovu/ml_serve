# Used by "mix format"
[
  inputs: [
    "{mix,.formatter}.exs",
    "{config,lib,test}/**/*.{ex,exs}",
    # The examples are compiled and run in CI, so they are held to the same formatting as lib.
    "examples/scripts/*.exs",
    "examples/inference_service/{mix,.formatter}.exs",
    "examples/inference_service/{config,lib,test}/**/*.{ex,exs}"
  ]
]
