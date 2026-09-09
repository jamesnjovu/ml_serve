defmodule MLServe.TelemetryTest do
  use MLServe.Case, async: true

  alias MLServe.Telemetry.Logger, as: TelemetryLogger
  alias MLServe.Telemetry.Metrics
  alias MLServe.Test.Backends

  describe "events/0" do
    test "lists every documented event" do
      events = MLServe.Telemetry.events()

      assert [:ml_serve, :prediction, :start] in events
      assert [:ml_serve, :prediction, :stop] in events
      assert [:ml_serve, :prediction, :exception] in events
      assert [:ml_serve, :model, :load] in events
      assert [:ml_serve, :model, :unload] in events
      assert [:ml_serve, :cache, :hit] in events
      assert [:ml_serve, :cache, :miss] in events
      assert [:ml_serve, :batch, :flush] in events
    end
  end

  describe "prediction span" do
    test "emits start then stop with the documented measurements" do
      ref = attach_telemetry([[:ml_serve, :prediction, :start], [:ml_serve, :prediction, :stop]])
      name = load!(backend: Backends.Echo)

      MLServe.predict(name, :x)

      {start_measurements, start_metadata} =
        assert_telemetry(ref, [:ml_serve, :prediction, :start], name)

      assert is_integer(start_measurements.system_time)
      assert is_integer(start_measurements.monotonic_time)
      assert start_measurements.batch_size == 1
      assert start_metadata.backend == Backends.Echo
      assert start_metadata.version == "1.0.0"
      refute start_metadata.batch?

      {stop_measurements, stop_metadata} =
        assert_telemetry(ref, [:ml_serve, :prediction, :stop], name)

      assert stop_measurements.duration > 0
      assert is_integer(stop_measurements.queue_duration)
      assert is_integer(stop_measurements.inference_duration)
      assert stop_metadata.result == :ok
      refute stop_metadata.cached?
    end

    test "reports an error result without an exception event" do
      ref =
        attach_telemetry([[:ml_serve, :prediction, :stop], [:ml_serve, :prediction, :exception]])

      name =
        load!(
          backend: MLServe.Backend.Static,
          config: [error: :unavailable]
        )

      assert MLServe.predict(name, :x) == {:error, :unavailable}

      # A backend that *returns* an error is an expected rejection, not an exception.
      {_measurements, metadata} = assert_telemetry(ref, [:ml_serve, :prediction, :stop], name)
      assert metadata.result == :error

      refute_received {[:ml_serve, :prediction, :exception], ^ref, _, %{model: ^name}}
    end

    test "a raising hook produces an exception event" do
      ref = attach_telemetry([[:ml_serve, :prediction, :exception]])
      name = load!(backend: Backends.Echo, preprocess: fn _ -> raise "boom" end)

      assert_raise RuntimeError, fn -> MLServe.predict(name, 1) end

      {measurements, metadata} = assert_telemetry(ref, [:ml_serve, :prediction, :exception], name)

      assert measurements.duration > 0
      assert metadata.kind == :error
      assert %RuntimeError{message: "boom"} = metadata.reason
      assert metadata.stacktrace != []
    end

    test "records queue and inference durations separately for pooled models" do
      ref = attach_telemetry([[:ml_serve, :prediction, :stop]])
      name = load!(backend: Backends.Slow, workers: 1, config: [delay: 30], timeout: 5_000)

      MLServe.predict(name, :x)

      {measurements, _metadata} =
        assert_telemetry(ref, [:ml_serve, :prediction, :stop], name, 2_000)

      assert measurements.inference_duration > 0
      assert measurements.duration >= measurements.inference_duration
    end

    test "batch predictions carry the batch size and batch? flag" do
      ref = attach_telemetry([[:ml_serve, :prediction, :stop]])
      name = load!(backend: Backends.Echo)

      MLServe.batch_predict(name, [1, 2, 3, 4])

      {measurements, metadata} = assert_telemetry(ref, [:ml_serve, :prediction, :stop], name)

      assert measurements.batch_size == 4
      assert metadata.batch?
    end
  end

  describe "model lifecycle events" do
    test "load emits a single event with a duration" do
      ref = attach_telemetry([[:ml_serve, :model, :load]])
      name = load!(backend: Backends.Pooled, workers: 2)

      {measurements, metadata} = assert_telemetry(ref, [:ml_serve, :model, :load], name, 2_000)

      assert measurements.duration > 0
      assert metadata.version == "1.0.0"
      assert metadata.backend == Backends.Pooled
      assert metadata.workers == 2
      assert metadata.result == :ok
    end

    test "a failed load emits an error result" do
      ref = attach_telemetry([[:ml_serve, :model, :load]])
      name = register!(backend: Backends.FailingLoad, config: [token: token()])

      {_measurements, metadata} = assert_telemetry(ref, [:ml_serve, :model, :load], name, 3_000)

      assert metadata.result == :error
    end

    test "unload emits a duration and a clean drain count" do
      ref = attach_telemetry([[:ml_serve, :model, :unload]])
      name = load!(backend: Backends.Echo)

      :ok = MLServe.unload_model(name)

      {measurements, metadata} = assert_telemetry(ref, [:ml_serve, :model, :unload], name, 2_000)

      assert is_integer(measurements.duration)
      assert measurements.drained == 0
      assert metadata.backend == Backends.Echo
    end
  end

  describe "MLServe.Telemetry.Logger" do
    test "attaching and detaching are idempotent" do
      assert :ok = TelemetryLogger.attach(level: :debug)
      assert :ok = TelemetryLogger.attach(level: :debug)
      assert :ok = TelemetryLogger.detach()
      assert :ok = TelemetryLogger.detach()
    end

    test "logs prediction events without raising" do
      :ok = TelemetryLogger.attach(level: :debug)
      on_exit(&TelemetryLogger.detach/0)

      name = load!(backend: Backends.Echo)

      assert MLServe.predict(name, :x) == {:ok, :x}
    end
  end

  describe "MLServe.Telemetry.Metrics" do
    test "returns Telemetry.Metrics definitions when the dependency is available" do
      metrics = Metrics.metrics()

      assert is_list(metrics)

      if Code.ensure_loaded?(Telemetry.Metrics) do
        assert metrics != []
        names = Enum.map(metrics, & &1.name)
        assert [:ml_serve, :prediction, :duration] in names
      end
    end

    test "every metric carries the model and version tags" do
      for metric <- Metrics.metrics() do
        assert :model in metric.tags, "#{inspect(metric.name)} is missing the :model tag"
      end
    end
  end
end
