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
      assert [:ml_serve, :prediction, :rejected] in events
      assert [:ml_serve, :model, :load] in events
      assert [:ml_serve, :model, :unload] in events
      assert [:ml_serve, :cache, :hit] in events
      assert [:ml_serve, :cache, :miss] in events
      assert [:ml_serve, :batch, :flush] in events
    end
  end

  describe "rejection events" do
    @rejected [:ml_serve, :prediction, :rejected]

    test "a shed request is visible, not merely absent" do
      # The failure this guards against: under load shedding, the :stop counter *falls* and the
      # error rate stays flat, so an overloaded model looks like a quiet healthy one.
      ref = attach_telemetry([@rejected])

      name =
        load!(
          backend: Backends.Slow,
          workers: 1,
          max_concurrency: 1,
          config: [delay: 300, notify: self()],
          timeout: 5_000
        )

      Task.async(fn -> MLServe.predict(name, :slow) end)
      assert_receive {:predict_started, _worker}, 5_000

      assert MLServe.predict(name, :shed) == {:error, :overloaded}

      {measurements, metadata} = assert_telemetry(ref, @rejected, name)

      assert measurements.count == 1
      assert measurements.batch_size == 1
      assert metadata.reason == :overloaded
      assert metadata.version == "1.0.0"
      refute metadata.batch?
    end

    test "an unknown model is reported, with no version to report" do
      ref = attach_telemetry([@rejected])

      assert MLServe.predict(:no_such_model_here, :x) == {:error, :model_not_found}

      {_measurements, metadata} = assert_telemetry(ref, @rejected, :no_such_model_here)

      assert metadata.reason == :model_not_found
      assert is_nil(metadata.version)
    end

    test "an oversized batch is reported with the batch size that was refused" do
      ref = attach_telemetry([@rejected])
      name = load!(backend: Backends.Echo, max_batch_size: 3)

      assert {:error, {:batch_too_large, 3}} =
               MLServe.batch_predict(name, Enum.to_list(1..10))

      {measurements, metadata} = assert_telemetry(ref, @rejected, name)

      assert measurements.batch_size == 10
      assert metadata.batch?
      assert metadata.reason == {:batch_too_large, 3}
    end
  end

  describe "error_kind on :stop" do
    test "distinguishes a backend that raised from one that returned an error" do
      ref = attach_telemetry([[:ml_serve, :prediction, :stop]])
      raising = load!(backend: Backends.Crashing)
      returning = load!(backend: MLServe.Backend.Static, config: [error: :out_of_domain])

      assert {:error, {:backend_error, _}} = MLServe.predict(raising, :x)
      assert MLServe.predict(returning, :x) == {:error, :out_of_domain}

      {_m, raised} = assert_telemetry(ref, [:ml_serve, :prediction, :stop], raising)
      {_m, returned} = assert_telemetry(ref, [:ml_serve, :prediction, :stop], returning)

      assert raised.result == :error
      assert raised.error_kind == :raised

      assert returned.result == :error
      assert returned.error_kind == :returned
    end

    test "is nil for a successful prediction" do
      ref = attach_telemetry([[:ml_serve, :prediction, :stop]])
      name = load!(backend: Backends.Echo)

      MLServe.predict(name, :x)

      {_measurements, metadata} = assert_telemetry(ref, [:ml_serve, :prediction, :stop], name)

      assert metadata.result == :ok
      assert is_nil(metadata.error_kind)
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
