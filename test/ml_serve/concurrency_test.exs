defmodule MLServe.ConcurrencyTest do
  use MLServe.Case, async: true

  alias MLServe.Test.Backends

  describe "concurrent predictions" do
    test "1000 concurrent requests all succeed across a pool of 4" do
      name = load!(backend: Backends.Pooled, workers: 4, timeout: 5_000)

      results =
        1..1_000
        |> Task.async_stream(fn i -> MLServe.predict(name, i) end,
          max_concurrency: 50,
          timeout: 10_000
        )
        |> Enum.map(fn {:ok, result} -> result end)

      assert length(results) == 1_000
      assert Enum.all?(results, &match?({:ok, {_input, _pid}}, &1))

      inputs = Enum.map(results, fn {:ok, {input, _pid}} -> input end)
      assert Enum.sort(inputs) == Enum.to_list(1..1_000)

      assert {:ok, %{requests: 1_000, errors: 0, in_flight: 0}} = MLServe.model_status(name)
    end

    test "round-robin spreads work evenly across workers" do
      name = load!(backend: Backends.Pooled, workers: 4, timeout: 5_000)

      counts =
        1..400
        |> Enum.map(fn i ->
          {:ok, {_input, pid}} = MLServe.predict(name, i)
          pid
        end)
        |> Enum.frequencies()
        |> Map.values()

      assert length(counts) == 4
      # Round-robin from a single caller is exact, not merely balanced.
      assert Enum.all?(counts, &(&1 == 100))
    end

    test "random selection uses every worker" do
      name = load!(backend: Backends.Pooled, workers: 4, selection: :random, timeout: 5_000)

      distinct =
        1..400
        |> Enum.map(fn i ->
          {:ok, {_input, pid}} = MLServe.predict(name, i)
          pid
        end)
        |> Enum.uniq()

      assert length(distinct) == 4
    end

    test "least_loaded selection serves every request" do
      name =
        load!(backend: Backends.Pooled, workers: 4, selection: :least_loaded, timeout: 5_000)

      results =
        1..200
        |> Task.async_stream(fn i -> MLServe.predict(name, i) end, max_concurrency: 20)
        |> Enum.map(fn {:ok, result} -> result end)

      assert Enum.all?(results, &match?({:ok, _}, &1))
    end

    test "shared backends serve concurrent requests without any worker process" do
      name = load!(backend: Backends.Caller, workers: 8)

      pids =
        1..200
        |> Task.async_stream(
          fn _ ->
            {:ok, pid} = MLServe.predict(name, nil)
            pid
          end,
          max_concurrency: 20
        )
        |> Enum.map(fn {:ok, pid} -> pid end)
        |> Enum.uniq()

      # Every prediction ran in its own caller. There is no pool to be a bottleneck.
      assert length(pids) > 1
      assert {:ok, %{workers: 0}} = MLServe.model_status(name)
    end

    test "a slow model does not block an unrelated model" do
      slow = load!(backend: Backends.Slow, workers: 1, config: [delay: 300], timeout: 5_000)
      fast = load!(backend: Backends.Echo)

      blockers =
        for i <- 1..4, do: Task.async(fn -> MLServe.predict(slow, i, timeout: 5_000) end)

      eventually(fn -> match?({:ok, %{in_flight: n}} when n > 0, MLServe.model_status(slow)) end)

      # The fast model is untouched by the queue building up on the slow one.
      started = System.monotonic_time(:millisecond)
      assert {:ok, :quick} = MLServe.predict(fast, :quick)
      elapsed = System.monotonic_time(:millisecond) - started

      assert elapsed < 100

      Enum.each(blockers, &Task.await(&1, 5_000))
    end

    test "counters stay consistent under concurrent load" do
      name = load!(backend: Backends.Pooled, workers: 4, timeout: 5_000)

      1..500
      |> Task.async_stream(fn i -> MLServe.predict(name, i) end, max_concurrency: 50)
      |> Stream.run()

      assert {:ok, %{requests: 500, errors: 0, in_flight: 0}} = MLServe.model_status(name)
    end
  end

  describe "per-worker loading" do
    test "load: :per_worker gives each worker its own state" do
      tok = token()
      name = load!(backend: Backends.PerWorker, workers: 4, config: [token: tok])

      instances =
        1..200
        |> Enum.map(fn _ ->
          {:ok, instance} = MLServe.predict(name, nil)
          instance
        end)
        |> Enum.uniq()
        |> Enum.sort()

      assert instances == [1, 2, 3, 4]
      assert Backends.count(tok, :load) == 4
    end

    test "load: :once loads the backend exactly once for the whole pool" do
      tok = token()
      name = load!(backend: Backends.Counting, workers: 4, config: [token: tok])

      for i <- 1..20, do: MLServe.predict(name, i)

      # Counting loads with :once by default, so state is shared across the pool. A 2GB model
      # loaded per worker would be an OOM, not a pool.
      assert Backends.count(tok, :predict) == 20
      assert {:ok, %{workers: 4}} = MLServe.model_status(name)
    end
  end

  describe "worker failure and supervision" do
    test "a crashed worker is replaced and predictions keep succeeding" do
      name = load!(backend: Backends.Pooled, workers: 3, timeout: 5_000)
      victim = MLServe.Worker.whereis(name, "1.0.0", 1)
      ref = Process.monitor(victim)

      Process.exit(victim, :kill)
      assert_receive {:DOWN, ^ref, :process, ^victim, :killed}, 1_000

      replacement =
        eventually(fn ->
          case MLServe.Worker.whereis(name, "1.0.0", 1) do
            nil -> nil
            ^victim -> nil
            pid -> pid
          end
        end)

      assert replacement != victim

      results = for i <- 1..30, do: MLServe.predict(name, i)
      assert Enum.all?(results, &match?({:ok, _}, &1))
    end

    test "requests are served while a worker is missing" do
      name = load!(backend: Backends.Pooled, workers: 3, timeout: 5_000)
      victim = MLServe.Worker.whereis(name, "1.0.0", 0)

      Process.exit(victim, :kill)

      # A killed worker stays in the Registry until its monitor fires, so dispatch can select a
      # pid that is already dead. Selection checks liveness and the caller retries a different
      # worker, which is what makes a restart invisible rather than a burst of failures.
      results = for i <- 1..30, do: MLServe.predict(name, i)

      assert Enum.all?(results, &match?({:ok, _}, &1)),
             "a request failed while a worker was restarting: #{inspect(Enum.reject(results, &match?({:ok, _}, &1)))}"
    end

    test "requests survive repeated worker kills" do
      name = load!(backend: Backends.Pooled, workers: 3, timeout: 5_000)

      killer =
        Task.async(fn ->
          for _ <- 1..15 do
            index = :rand.uniform(3) - 1

            case MLServe.Worker.whereis(name, "1.0.0", index) do
              nil -> :ok
              pid -> Process.exit(pid, :kill)
            end

            Process.sleep(5)
          end
        end)

      results =
        1..300
        |> Task.async_stream(fn i -> MLServe.predict(name, i) end, max_concurrency: 10)
        |> Enum.map(fn {:ok, result} -> result end)

      Task.await(killer, 5_000)

      failures = Enum.reject(results, &match?({:ok, _}, &1))

      assert failures == [],
             "#{length(failures)} of 300 requests failed: #{inspect(Enum.take(failures, 3))}"
    end

    test "a worker killed mid-request returns an error instead of killing the caller" do
      name = load!(backend: Backends.Slow, workers: 1, config: [delay: 300], timeout: 5_000)

      caller =
        Task.async(fn ->
          Process.flag(:trap_exit, true)
          MLServe.predict(name, :slow, timeout: 5_000)
        end)

      eventually(fn -> match?({:ok, %{in_flight: 1}}, MLServe.model_status(name)) end)
      Process.exit(MLServe.Worker.whereis(name, "1.0.0", 0), :kill)

      # The caller must survive. A GenServer.call to a worker that is killed while serving exits
      # the *caller* with :killed unless every exit is caught — which would take a Phoenix request
      # process down with the worker and destroy the isolation this library exists to provide.
      result = Task.await(caller, 5_000)

      assert result == {:error, :model_not_ready}
    end

    test "a crashed model server restarts the whole subtree and reloads the model" do
      name = load!(backend: Backends.Pooled, workers: 2, timeout: 5_000)
      server = MLServe.ModelServer.whereis(name, "1.0.0")
      worker = MLServe.Worker.whereis(name, "1.0.0", 0)
      ref = Process.monitor(server)

      Process.exit(server, :kill)
      assert_receive {:DOWN, ^ref, :process, ^server, :killed}, 1_000

      # :rest_for_one means the workers restart behind the server, because the state they hold
      # came from it and is now stale.
      new_worker =
        eventually(fn ->
          case MLServe.Worker.whereis(name, "1.0.0", 0) do
            nil -> nil
            ^worker -> nil
            pid -> pid
          end
        end)

      assert new_worker != worker
      assert :ok = MLServe.await_ready(name, 2_000)
      assert {:ok, {5, _pid}} = MLServe.predict(name, 5)
    end

    test "one model crashing does not disturb another" do
      crashing = load!(backend: Backends.Pooled, workers: 1, timeout: 5_000)
      healthy = load!(backend: Backends.Echo)
      healthy_server = MLServe.ModelServer.whereis(healthy, "1.0.0")

      server = MLServe.ModelServer.whereis(crashing, "1.0.0")
      Process.exit(server, :kill)

      assert MLServe.predict(healthy, :fine) == {:ok, :fine}
      assert MLServe.ModelServer.whereis(healthy, "1.0.0") == healthy_server
    end
  end
end
