defmodule InferenceService.Application do
  @moduledoc """
  Supervision tree for the service.

  MLServe is not started here. It is an OTP application in its own right and boots itself,
  including every model declared in `config/config.exs`. This tree only owns the HTTP endpoint
  and the telemetry handlers.
  """

  use Application

  require Logger

  @impl true
  def start(_type, _args) do
    port = Application.get_env(:inference_service, :port, 4000)

    InferenceService.Telemetry.attach()

    children = [
      {Bandit, plug: InferenceService.Router, port: port}
    ]

    Logger.info("inference service listening on http://localhost:#{port}")

    Supervisor.start_link(children, strategy: :one_for_one, name: InferenceService.Supervisor)
  end
end
