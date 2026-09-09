defmodule MLServe.Bootstrap do
  @moduledoc """
  Loads the models declared in application configuration when MLServe starts.

  A transient one-shot process rather than logic inside the application callback. Loading
  models from `start/2` would mean a bad model configuration prevents the whole application from
  booting — including the parts that have nothing to do with inference, and including the
  health-check endpoint that would tell you why.

  In `:async` mode (the default) it registers each model and returns immediately; models load in
  the background and report through `MLServe.model_status/2`. In `:sync` mode it waits for every
  model to become ready or fail, which is what you want in tests and short-lived scripts.
  """

  use Task, restart: :transient

  require Logger

  @doc false
  @spec start_link(keyword()) :: {:ok, pid()}
  def start_link(opts) do
    Task.start_link(__MODULE__, :run, [opts])
  end

  @doc false
  @spec run(keyword()) :: :ok
  def run(opts) do
    models = Keyword.get(opts, :models) || MLServe.Config.configured_models()
    mode = Keyword.get(opts, :start_mode) || MLServe.Config.start_mode()

    loaded =
      for {name, model_opts} <- models, reduce: [] do
        acc ->
          case MLServe.load_model(name, model_opts) do
            {:ok, _ref} ->
              [name | acc]

            {:error, reason} ->
              Logger.error(
                "[ml_serve] failed to register model #{inspect(name)}: #{inspect(reason)}"
              )

              acc
          end
      end

    if mode == :sync do
      Enum.each(loaded, &MLServe.await_ready(&1, 30_000))
    end

    :ok
  end
end
