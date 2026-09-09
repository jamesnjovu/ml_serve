defmodule MLServe.Telemetry.Logger do
  @moduledoc """
  Logs MLServe telemetry events, with no dependencies beyond `Logger`.

  The fastest way to see what MLServe is doing. Attach it in `application.ex` while developing,
  or in production at `:info` to record model loads and unloads without wiring up a metrics
  backend:

      MLServe.Telemetry.Logger.attach(level: :info, events: [:model])

  For real observability use `MLServe.Telemetry.Metrics` with LiveDashboard or a StatsD/Prometheus
  reporter — logging every prediction in production is a way to spend your I/O budget on strings.

  ## Options

    * `:level` — log level, defaulting to `:debug`
    * `:events` — which groups to log: `:prediction`, `:model`, `:cache`, `:batch`.
      Defaults to `[:prediction, :model]`; cache and batch events are high-volume.
  """

  require Logger

  @handler_id {__MODULE__, :handler}

  @groups %{
    prediction: [
      [:ml_serve, :prediction, :stop],
      [:ml_serve, :prediction, :exception]
    ],
    model: [
      [:ml_serve, :model, :load],
      [:ml_serve, :model, :unload]
    ],
    cache: [
      [:ml_serve, :cache, :hit],
      [:ml_serve, :cache, :miss]
    ],
    batch: [
      [:ml_serve, :batch, :flush]
    ]
  }

  @doc """
  Attaches the logger. Safe to call more than once.

  ## Examples

      MLServe.Telemetry.Logger.attach()
      MLServe.Telemetry.Logger.attach(level: :info, events: [:model])
  """
  @spec attach(keyword()) :: :ok
  def attach(opts \\ []) do
    level = Keyword.get(opts, :level, :debug)
    groups = Keyword.get(opts, :events, [:prediction, :model])
    events = Enum.flat_map(groups, &Map.get(@groups, &1, []))

    detach()
    :telemetry.attach_many(@handler_id, events, &__MODULE__.handle_event/4, %{level: level})
  end

  @doc "Detaches the logger. Safe to call when not attached."
  @spec detach() :: :ok
  def detach do
    :telemetry.detach(@handler_id)
    :ok
  end

  @doc false
  @spec handle_event([atom()], map(), map(), map()) :: :ok
  def handle_event(event, measurements, metadata, %{level: level}) do
    Logger.log(level, fn -> format(event, measurements, metadata) end)
  end

  # Private Functions

  defp format([:ml_serve, :prediction, :stop], measurements, metadata) do
    "[ml_serve] #{model(metadata)} #{metadata.result} in #{ms(measurements.duration)}ms" <>
      " (inference #{ms(measurements.inference_duration)}ms, queue #{ms(measurements.queue_duration)}ms" <>
      ", batch #{measurements.batch_size}#{cached(metadata)}#{canary(metadata)})"
  end

  defp format([:ml_serve, :prediction, :exception], measurements, metadata) do
    "[ml_serve] #{model(metadata)} raised after #{ms(measurements.duration)}ms: " <>
      Exception.format_banner(metadata.kind, metadata.reason, [])
  end

  defp format([:ml_serve, :model, :load], measurements, metadata) do
    "[ml_serve] loaded #{model(metadata)} (#{metadata.result}) in #{ms(measurements.duration)}ms" <>
      " with #{metadata.workers} worker(s)"
  end

  defp format([:ml_serve, :model, :unload], measurements, metadata) do
    drained =
      case measurements.drained do
        0 -> "drained cleanly"
        n -> "abandoned #{n} in-flight request(s)"
      end

    "[ml_serve] unloaded #{model(metadata)} in #{ms(measurements.duration)}ms, #{drained}"
  end

  defp format([:ml_serve, :cache, outcome], _measurements, metadata) do
    "[ml_serve] cache #{outcome} for #{model(metadata)}"
  end

  defp format([:ml_serve, :batch, :flush], measurements, metadata) do
    "[ml_serve] #{model(metadata)} flushed #{measurements.size} input(s) on #{metadata.reason}" <>
      " after #{ms(measurements.wait_duration)}ms"
  end

  defp model(%{model: model, version: version}), do: "#{inspect(model)} v#{version}"
  defp model(%{model: model}), do: inspect(model)

  defp cached(%{cached?: true}), do: ", cached"
  defp cached(_metadata), do: ""

  defp canary(%{canary?: true}), do: ", canary"
  defp canary(_metadata), do: ""

  defp ms(native) do
    native
    |> System.convert_time_unit(:native, :microsecond)
    |> Kernel./(1000)
    |> Float.round(2)
  end
end
