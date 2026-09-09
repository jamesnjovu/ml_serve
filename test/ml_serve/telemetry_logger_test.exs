defmodule MLServe.TelemetryLoggerTest do
  # Not async: attaching a telemetry handler is global, and capture_log/1 captures every process's
  # output. A concurrent test's predictions would land in this test's captured log.
  use MLServe.Case, async: false

  import ExUnit.CaptureLog

  alias MLServe.Telemetry.Logger, as: TelemetryLogger
  alias MLServe.Test.Backends

  setup do
    on_exit(&TelemetryLogger.detach/0)
    :ok
  end

  # Attach, capture, detach — in that order. Detaching inside the test body rather than in on_exit
  # matters: on_exit callbacks run in reverse registration order, so the model unload registered by
  # load!/2 would otherwise fire while the handler is still attached and log over the suite output.
  defp capture_ml_log(attach_opts \\ [], fun) do
    :ok = TelemetryLogger.attach(Keyword.put_new(attach_opts, :level, :warning))
    log = capture_log(fun)
    :ok = TelemetryLogger.detach()
    log
  end

  describe "prediction logging" do
    test "formats a successful prediction with its timings" do
      name = load!(backend: Backends.Echo)

      log = capture_ml_log(fn -> MLServe.predict(name, :x) end)

      assert log =~ "[ml_serve] #{inspect(name)} v1.0.0 ok in"
      assert log =~ "inference"
      assert log =~ "queue"
      assert log =~ "batch 1"
    end

    test "marks a cached result" do
      name = load!(backend: Backends.Echo)
      MLServe.predict(name, :x, cache: true)

      log = capture_ml_log(fn -> MLServe.predict(name, :x, cache: true) end)

      assert log =~ "cached"
    end

    test "marks canary traffic" do
      name = load!(backend: MLServe.Backend.Static, config: [result: :v1])

      {:ok, _} =
        MLServe.load_model(name,
          backend: MLServe.Backend.Static,
          version: "2.0.0",
          config: [result: :v2]
        )

      :ok = MLServe.await_ready({name, "2.0.0"})
      :ok = MLServe.canary(name, "2.0.0", 100)

      log = capture_ml_log(fn -> MLServe.predict(name, nil) end)

      assert log =~ "v2.0.0"
      assert log =~ "canary"
    end

    test "reports an error result" do
      name = load!(backend: MLServe.Backend.Static, config: [error: :unavailable])

      log = capture_ml_log(fn -> MLServe.predict(name, :x) end)

      assert log =~ "error in"
    end

    test "formats a backend exception with its banner" do
      name = load!(backend: Backends.Echo, preprocess: fn _ -> raise "kaboom" end)

      log =
        capture_ml_log(fn ->
          assert_raise RuntimeError, fn -> MLServe.predict(name, 1) end
        end)

      assert log =~ "raised after"
      assert log =~ "kaboom"
    end

    test "reports the batch size" do
      name = load!(backend: Backends.Echo)

      log = capture_ml_log(fn -> MLServe.batch_predict(name, [1, 2, 3, 4, 5]) end)

      assert log =~ "batch 5"
    end
  end

  describe "lifecycle logging" do
    test "formats a load with the worker count" do
      log =
        capture_ml_log([events: [:model]], fn -> load!(backend: Backends.Pooled, workers: 2) end)

      assert log =~ "loaded"
      assert log =~ "(ok)"
      assert log =~ "with 2 worker(s)"
    end

    test "formats a clean unload" do
      name = load!(backend: Backends.Echo)

      log = capture_ml_log([events: [:model]], fn -> MLServe.unload_model(name) end)

      assert log =~ "unloaded"
      assert log =~ "drained cleanly"
    end

    test "reports abandoned requests when a drain times out" do
      name = load!(backend: Backends.Slow, workers: 1, config: [delay: 400], timeout: 2_000)

      task = Task.async(fn -> MLServe.predict(name, :slow) end)
      eventually(fn -> match?({:ok, %{in_flight: 1}}, MLServe.model_status(name)) end)

      log = capture_ml_log([events: [:model]], fn -> MLServe.unload_model(name, timeout: 50) end)

      assert log =~ "abandoned 1 in-flight request(s)"
      Task.shutdown(task, :brutal_kill)
    end
  end

  describe "cache and batch logging" do
    test "logs cache hits and misses when enabled" do
      name = load!(backend: Backends.Echo)

      log =
        capture_ml_log([events: [:cache]], fn ->
          MLServe.predict(name, :x, cache: true)
          MLServe.predict(name, :x, cache: true)
        end)

      assert log =~ "cache miss"
      assert log =~ "cache hit"
    end

    test "logs a batch flush with its reason" do
      name =
        load!(
          backend: Backends.Counting,
          workers: 1,
          config: [token: token()],
          batching: [max_size: 50, timeout: 10],
          timeout: 5_000
        )

      log = capture_ml_log([events: [:batch]], fn -> MLServe.predict(name, 1) end)

      assert log =~ "flushed 1 input(s) on timeout"
    end
  end

  describe "attach/1" do
    test "defaults to prediction and model events only" do
      name = load!(backend: Backends.Echo)

      # Cache events are high-volume, so they are off unless asked for.
      log = capture_ml_log(fn -> MLServe.predict(name, :x, cache: true) end)

      refute log =~ "cache miss"
      assert log =~ "ok in"
    end

    test "re-attaching does not double-log" do
      name = load!(backend: Backends.Echo)
      :ok = TelemetryLogger.attach(level: :warning)

      log = capture_ml_log(fn -> MLServe.predict(name, :x) end)

      assert length(String.split(log, "[ml_serve] #{inspect(name)}")) == 2
    end

    test "detaching stops logging" do
      name = load!(backend: Backends.Echo)
      :ok = TelemetryLogger.attach(level: :warning)
      :ok = TelemetryLogger.detach()

      refute capture_log(fn -> MLServe.predict(name, :x) end) =~ "[ml_serve]"
    end
  end
end
