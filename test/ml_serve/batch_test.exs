defmodule MLServe.BatchTest do
  use MLServe.Case, async: true

  alias MLServe.Test.Backends

  describe "batch_predict/3 with a native batching backend" do
    test "makes exactly one backend call for the whole batch" do
      tok = token()
      name = load!(backend: Backends.Counting, workers: 1, config: [token: tok])

      assert {:ok, results} = MLServe.batch_predict(name, [1, 2, 3, 4, 5])

      assert results == [
               {:batched, 1},
               {:batched, 2},
               {:batched, 3},
               {:batched, 4},
               {:batched, 5}
             ]

      # The whole point: one round-trip, not five.
      assert Backends.count(tok, :batch) == 1
      assert Backends.count(tok, :batch_rows) == 5
      assert Backends.count(tok, :predict) == 0
    end

    test "preserves input order" do
      tok = token()
      name = load!(backend: Backends.Counting, workers: 2, config: [token: tok])

      inputs = Enum.to_list(1..50)
      assert {:ok, results} = MLServe.batch_predict(name, inputs)
      assert results == Enum.map(inputs, &{:batched, &1})
    end
  end

  describe "batch_predict/3 without a native batching backend" do
    test "falls back to mapping predict over the inputs" do
      name = load!(backend: Backends.Echo)

      assert MLServe.batch_predict(name, [1, 2, 3]) == {:ok, [1, 2, 3]}
    end

    test "a backend that exports batch_predict but returns :not_supported falls back too" do
      # MLServe.Backend.Function exports batch_predict/2 but only batches when configured to.
      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [predict: fn n -> {:ok, n * 2} end]
        )

      assert MLServe.batch_predict(name, [1, 2, 3]) == {:ok, [2, 4, 6]}
    end

    test "stops at the first error rather than running the remaining inputs" do
      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [
            predict: fn
              3 -> {:error, :bad_row}
              n -> {:ok, n}
            end
          ]
        )

      assert MLServe.batch_predict(name, [1, 2, 3, 4]) == {:error, :bad_row}
    end
  end

  describe "batch validation" do
    test "rejects a batch larger than max_batch_size" do
      name = load!(backend: Backends.Echo, max_batch_size: 3)

      assert MLServe.batch_predict(name, [1, 2, 3, 4]) == {:error, {:batch_too_large, 3}}
      assert MLServe.batch_predict(name, [1, 2, 3]) == {:ok, [1, 2, 3]}
    end

    test "rejects a non-list input" do
      name = load!(backend: Backends.Echo)

      assert {:error, {:invalid_input, message}} = MLServe.batch_predict(name, :not_a_list)
      assert message =~ "expects a list"
    end

    test "an empty batch is valid" do
      name = load!(backend: Backends.Echo)

      assert MLServe.batch_predict(name, []) == {:ok, []}
    end

    test "rejects a backend returning the wrong number of results" do
      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [
            predict: fn n -> {:ok, n} end,
            batch_predict: fn _inputs -> [:only_one] end
          ]
        )

      assert {:error, {:backend_error, error}} = MLServe.batch_predict(name, [1, 2, 3])
      assert Exception.message(error) =~ "returned 1 results for 3 inputs"
    end
  end

  describe "hooks in batches" do
    test "preprocess and postprocess apply to every element" do
      name =
        load!(
          backend: Backends.Echo,
          preprocess: fn n -> {:ok, n + 1} end,
          postprocess: fn n -> {:ok, n * 10} end
        )

      assert MLServe.batch_predict(name, [1, 2, 3]) == {:ok, [20, 30, 40]}
    end

    test "one invalid element rejects the whole batch" do
      name =
        load!(
          backend: Backends.Echo,
          preprocess: fn
            n when is_integer(n) -> {:ok, n}
            _ -> {:error, {:invalid_input, :not_an_integer}}
          end
        )

      assert MLServe.batch_predict(name, [1, :bad, 3]) ==
               {:error, {:invalid_input, :not_an_integer}}
    end
  end

  describe "dynamic batching" do
    test "coalesces concurrent single predictions into one backend call" do
      tok = token()

      name =
        load!(
          backend: Backends.Counting,
          workers: 1,
          config: [token: tok],
          batching: [max_size: 10, timeout: 50],
          timeout: 5_000
        )

      results =
        1..10
        |> Task.async_stream(fn i -> MLServe.predict(name, i) end, max_concurrency: 10)
        |> Enum.map(fn {:ok, result} -> result end)

      assert Enum.all?(results, &match?({:ok, {:batched, _}}, &1))

      returned = Enum.map(results, fn {:ok, {:batched, i}} -> i end)
      assert Enum.sort(returned) == Enum.to_list(1..10)

      # Ten independent callers, one backend invocation.
      assert Backends.count(tok, :batch) == 1
      assert Backends.count(tok, :batch_rows) == 10
    end

    test "flushes on the timeout when the batch never fills" do
      tok = token()

      name =
        load!(
          backend: Backends.Counting,
          workers: 1,
          config: [token: tok],
          batching: [max_size: 100, timeout: 20],
          timeout: 5_000
        )

      assert {:ok, {:batched, 1}} = MLServe.predict(name, 1)
      assert Backends.count(tok, :batch) == 1
    end

    test "emits a flush event describing why the batch closed" do
      ref = attach_telemetry([[:ml_serve, :batch, :flush]])

      name =
        load!(
          backend: Backends.Counting,
          workers: 1,
          config: [token: token()],
          batching: [max_size: 3, timeout: 500],
          timeout: 5_000
        )

      Task.async_stream(1..3, fn i -> MLServe.predict(name, i) end, max_concurrency: 3)
      |> Stream.run()

      {measurements, metadata} = assert_telemetry(ref, [:ml_serve, :batch, :flush], name, 1_000)

      assert measurements.size == 3
      assert metadata.reason == :full
    end

    test "reports a timeout flush distinctly from a full one" do
      ref = attach_telemetry([[:ml_serve, :batch, :flush]])

      name =
        load!(
          backend: Backends.Counting,
          workers: 1,
          config: [token: token()],
          batching: [max_size: 50, timeout: 20],
          timeout: 5_000
        )

      MLServe.predict(name, 1)

      assert {%{size: 1}, %{reason: :timeout}} =
               assert_telemetry(ref, [:ml_serve, :batch, :flush], name, 1_000)
    end

    test "the batcher keeps accepting work while a batch is in flight" do
      tok = token()

      name =
        load!(
          backend: Backends.Counting,
          workers: 2,
          config: [token: tok, delay: 60],
          batching: [max_size: 5, timeout: 10],
          timeout: 5_000
        )

      results =
        1..25
        |> Task.async_stream(fn i -> MLServe.predict(name, i) end,
          max_concurrency: 25,
          timeout: 5_000
        )
        |> Enum.map(fn {:ok, result} -> result end)

      assert Enum.all?(results, &match?({:ok, {:batched, _}}, &1))
      assert Backends.count(tok, :batch_rows) == 25
      # Batched into groups rather than served one at a time.
      assert Backends.count(tok, :batch) < 25
    end

    test "expired requests are dropped from the batch, not run" do
      tok = token()

      name =
        load!(
          backend: Backends.Counting,
          workers: 1,
          config: [token: tok, delay: 5],
          batching: [max_size: 50, timeout: 100],
          timeout: 5_000
        )

      assert MLServe.predict(name, 1, timeout: 10) == {:error, :timeout}
      Process.sleep(200)

      assert Backends.count(tok, :batch) == 0
    end

    test "a batch task that dies frees its slot and answers its callers" do
      name =
        load!(
          backend: Backends.Slow,
          workers: 1,
          config: [delay: 300],
          batching: [max_size: 2, timeout: 10],
          timeout: 5_000
        )

      batcher = Registry.lookup(MLServe.Registry, {:batcher, name, "1.0.0"}) |> hd() |> elem(0)

      callers =
        for _ <- 1..2, do: Task.async(fn -> MLServe.predict(name, :slow, timeout: 5_000) end)

      # Kill the flush task mid-batch. Without a monitor the in-flight slot leaks — wedging the
      # batcher permanently — and these callers block until their own timeouts.
      task =
        eventually(fn ->
          MLServe.TaskSupervisor |> Task.Supervisor.children() |> List.first()
        end)

      Process.exit(task, :kill)

      results = Enum.map(callers, &Task.await(&1, 5_000))
      assert Enum.all?(results, &match?({:error, _}, &1))

      # The batcher recovered its slot rather than deadlocking.
      assert Process.alive?(batcher)

      assert eventually(
               fn -> match?({:ok, _}, MLServe.predict(name, :later, timeout: 5_000)) end,
               3_000
             )
    end

    test "batch_predict/3 bypasses the batcher and calls the backend directly" do
      tok = token()

      name =
        load!(
          backend: Backends.Counting,
          workers: 1,
          config: [token: tok],
          batching: [max_size: 10, timeout: 50],
          timeout: 5_000
        )

      assert {:ok, results} = MLServe.batch_predict(name, [1, 2, 3])
      assert length(results) == 3
      assert Backends.count(tok, :batch) == 1
    end
  end
end
