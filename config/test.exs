import Config

# The suite deliberately exercises load failures, worker crashes and implicit default-version
# changes, all of which log. Only warnings and above are interesting when a test fails.
config :logger, level: :warning

# No models are declared here: every test registers its own under a unique name so the suite can
# run async against the singleton application. See MLServe.Case.
config :ml_serve, models: []
