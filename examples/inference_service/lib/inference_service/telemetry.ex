defmodule InferenceService.Telemetry do
  @moduledoc """
  Logs slow and failed predictions.

  A deployment would send these to StatsD, Prometheus or LiveDashboard instead — see
  `MLServe.Telemetry.Metrics.metrics/0`, which returns ready-made `Telemetry.Metrics`
  definitions tagged by `:model` and `:version`. This is the smallest thing that is still
  genuinely useful.
  """

  require Logger

  @slow_threshold_ms 500

  def attach do
    :telemetry.attach(
      "inference-service-predictions",
      [:ml_serve, :prediction, :stop],
      &__MODULE__.handle_event/4,
      nil
    )
  end

  def handle_event(_event, measurements, metadata, _config) do
    duration_ms = System.convert_time_unit(measurements.duration, :native, :millisecond)

    cond do
      metadata.result == :error ->
        Logger.warning("#{metadata.model} v#{metadata.version} failed in #{duration_ms}ms")

      duration_ms > @slow_threshold_ms ->
        queue_ms = System.convert_time_unit(measurements.queue_duration, :native, :millisecond)

        Logger.warning(
          "#{metadata.model} v#{metadata.version} took #{duration_ms}ms " <>
            "(#{queue_ms}ms queued) — the pool may be too small"
        )

      true ->
        :ok
    end
  end
end
