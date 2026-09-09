defmodule MLServe.MixProject do
  use Mix.Project

  @version "0.1.3"
  @source_url "https://github.com/jamesnjovu/ml_serve"

  def project do
    [
      app: :ml_serve,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      package: package(),
      name: "MLServe",
      source_url: @source_url,
      homepage_url: @source_url,
      description: description(),
      deps: deps(),
      docs: docs(),
      aliases: aliases(),
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.html": :test,
        "coveralls.github": :test
      ],
      dialyzer: [
        plt_add_apps: [:mix, :ex_unit],
        plt_file: {:no_warn, "priv/plts/dialyzer-#{Mix.env()}.plt"}
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger, :crypto],
      mod: {MLServe.Application, []}
    ]
  end

  # `test/support` holds the test-only model backends and the shared ExUnit case template.
  # Neither is compiled for consumers, who always build in :prod.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Shown on hex.pm and indexed by Hex search, Google, and LLM crawlers. Lead with the concrete
  # nouns people actually search for: "Elixir" and "machine learning", not "BEAM", which is
  # insider vocabulary that appears in almost none of the queries this needs to rank for.
  defp description do
    """
    Serve machine learning models in Elixir. Production ML inference for Phoenix and the BEAM:
    OTP supervision, worker pools, dynamic batching, caching, telemetry, model versioning and
    zero-downtime canary rollout around any backend — Nx, Bumblebee, ONNX, Python or a remote
    service.
    """
  end

  defp package do
    [
      name: "ml_serve",
      # Allowlist. `test/` and `priv/plts/` are excluded by omission.
      files: ~w(lib guides usage-rules.md llms.txt
                .formatter.exs mix.exs README.md CHANGELOG.md LICENSE),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Documentation" => "https://hexdocs.pm/ml_serve",
        "Changelog" => "#{@source_url}/blob/main/CHANGELOG.md",
        "Usage rules (for AI agents)" => "#{@source_url}/blob/main/usage-rules.md",
        "Issues" => "#{@source_url}/issues"
      },
      maintainers: ["James Njovu"]
    ]
  end

  defp deps do
    [
      # The only runtime dependency. Instrumentation is a first-class feature of this library and
      # :telemetry is the ecosystem-standard dispatcher — everything else stays optional.
      {:telemetry, "~> 1.0"},

      # Tooling
      # :test as well as :dev — the CI lint job runs with MIX_ENV=test, so a dev-only dialyxir
      # makes `mix dialyzer` an unknown task there.
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.40", only: :dev, runtime: false},
      {:excoveralls, "~> 0.18", only: :test, runtime: false},
      {:mix_audit, "~> 2.0", only: [:dev, :test], runtime: false},

      # Optional. MLServe.Telemetry.Metrics returns Telemetry.Metrics definitions when this is
      # present and an empty list when it is not, so LiveDashboard users get metrics for free
      # without every consumer paying for the dependency.
      {:telemetry_metrics, "~> 0.6 or ~> 1.0", optional: true}
    ]
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      source_url: @source_url,
      authors: ["James Njovu"],
      api_reference: true,
      extras: [
        "README.md": [title: "Overview"],
        "guides/getting-started.md": [title: "Getting Started"],
        "guides/architecture.md": [title: "Architecture"],
        "guides/creating-a-backend.md": [title: "Creating a Model Backend"],
        "guides/running-inference.md": [title: "Running Inference"],
        "guides/batch-inference.md": [title: "Batch Inference"],
        "guides/concurrency.md": [title: "Concurrency"],
        "guides/telemetry.md": [title: "Telemetry"],
        "guides/model-versioning.md": [title: "Model Versioning"],
        "guides/phoenix-integration.md": [title: "Phoenix Integration"],
        "guides/oban-integration.md": [title: "Oban Integration"],
        "guides/production-deployment.md": [title: "Production Deployment"],
        "CHANGELOG.md": [title: "Changelog"],
        LICENSE: [title: "License"]
      ],
      groups_for_extras: [
        Guides: Path.wildcard("guides/*.md")
      ],
      groups_for_modules: [
        Core: [MLServe, MLServe.Model, MLServe.Backend],
        Backends: ~r"MLServe.Backend\.",
        Observability: ~r"MLServe.Telemetry",
        Errors: [MLServe.Error, MLServe.BackendError],
        Runtime: [
          MLServe.Application,
          MLServe.Batcher,
          MLServe.Cache,
          MLServe.Config,
          MLServe.ModelInstance,
          MLServe.ModelRegistry,
          MLServe.ModelServer,
          MLServe.ModelSpec,
          MLServe.ModelSupervisor,
          MLServe.Security,
          MLServe.Supervisor,
          MLServe.Worker,
          MLServe.WorkerSupervisor
        ]
      ],
      skip_undefined_reference_warnings_on: ["CHANGELOG.md"]
    ]
  end

  defp aliases do
    [
      # One command for everything CI's lint job gates on, so a local run and a CI run cannot
      # disagree about whether the tree is clean.
      lint: [
        "format --check-formatted",
        "compile --warnings-as-errors",
        "credo --strict"
      ]
    ]
  end
end
