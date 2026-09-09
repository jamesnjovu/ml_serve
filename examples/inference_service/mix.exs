defmodule InferenceService.MixProject do
  use Mix.Project

  def project do
    [
      app: :inference_service,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {InferenceService.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Running from a clone of the repo. Against the published package this is:
      #   {:ml_serve, "~> 0.1.0"}
      {:ml_serve, path: "../.."},
      {:bandit, "~> 1.0"},
      {:plug, "~> 1.15"},
      {:jason, "~> 1.4"}
    ]
  end
end
