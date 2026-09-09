defmodule MLServe.Application do
  @moduledoc """
  The MLServe OTP application.

  Starts `MLServe.Supervisor`. Everything MLServe does — the catalog, the cache, every loaded
  model — lives under that tree, so a host application gets the whole runtime by depending on
  `:ml_serve` and nothing else.
  """

  use Application

  @impl true
  def start(_type, _args) do
    MLServe.Supervisor.start_link([])
  end
end
