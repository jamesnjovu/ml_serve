defmodule MLServe.ModelLifecycleTest do
  use MLServe.Case, async: true

  alias MLServe.Test.Backends

  describe "load_model/2" do
    test "registers a model and makes it ready" do
      name = load!(backend: Backends.Echo)

      assert {:ok, status} = MLServe.model_status(name)
      assert status.status == :ready
      assert status.name == name
      assert status.version == "1.0.0"
      assert status.backend == Backends.Echo
      assert status.default?
    end

    test "the first version registered becomes the default" do
      name = load!(backend: Backends.Echo, version: "3.2.1")

      assert MLServe.ModelRegistry.default_version(name) == {:ok, "3.2.1"}
    end

    test "records load duration and timestamp" do
      name = load!(backend: Backends.Echo)

      {:ok, status} = MLServe.model_status(name)
      assert is_integer(status.load_duration_ms)
      assert %DateTime{} = status.loaded_at
    end

    test "exposes backend metadata" do
      name = load!(backend: Backends.Echo, config: [tag: :custom])

      assert {:ok, %{metadata: %{tag: :custom}}} = MLServe.model_status(name)
    end

    test "refuses to load the same version twice" do
      name = load!(backend: Backends.Echo)

      assert {:error, {:already_loaded, ^name, "1.0.0"}} =
               MLServe.load_model(name, backend: Backends.Echo)
    end

    test "loads a second version alongside the first" do
      name = load!(backend: Backends.Echo)
      {:ok, _} = MLServe.load_model(name, backend: Backends.Echo, version: "2.0.0")
      :ok = MLServe.await_ready({name, "2.0.0"})

      assert {:ok, ["1.0.0", "2.0.0"]} = MLServe.versions(name)
      # The default does not move on its own — promotion is explicit.
      assert MLServe.ModelRegistry.default_version(name) == {:ok, "1.0.0"}
    end

    test "starts a worker process per worker for exclusive backends" do
      name = load!(backend: Backends.Pooled, workers: 3)

      assert {:ok, %{workers: 3, concurrency: :exclusive}} = MLServe.model_status(name)
      assert Enum.all?(0..2, &(MLServe.Worker.whereis(name, "1.0.0", &1) != nil))
    end

    test "starts no worker processes for shared backends" do
      name = load!(backend: Backends.Echo, workers: 8)

      assert {:ok, %{workers: 0, concurrency: :shared}} = MLServe.model_status(name)
      assert MLServe.Worker.whereis(name, "1.0.0", 0) == nil
    end

    test "marks the model failed after exhausting load retries" do
      tok = token()
      name = register!(backend: Backends.FailingLoad, config: [token: tok])

      eventually(fn ->
        match?({:ok, %{status: :failed}}, MLServe.model_status(name))
      end)

      assert {:ok, status} = MLServe.model_status(name)
      assert status.failure == :deliberate_load_failure
      # Retried with backoff rather than crash-looping the supervisor.
      assert Backends.count(tok, :load) == 3
    end

    test "recovers when a retried load eventually succeeds" do
      tok = token()

      name =
        register!(backend: Backends.FailingLoad, config: [token: tok, succeed_after: 1])

      assert :ok = MLServe.await_ready(name, 3_000)
      assert {:ok, %{status: :ready}} = MLServe.model_status(name)
      assert MLServe.predict(name, :anything) == {:ok, :recovered}
    end

    test "a failed model is not routable" do
      name = register!(backend: Backends.FailingLoad, config: [token: token()])

      eventually(fn -> match?({:ok, %{status: :failed}}, MLServe.model_status(name)) end)

      assert MLServe.predict(name, 1) == {:error, :model_not_ready}
      refute MLServe.ready?(name)
    end
  end

  describe "unload_model/2" do
    test "removes the model" do
      name = load!(backend: Backends.Echo)

      assert :ok = MLServe.unload_model(name)
      assert MLServe.model_status(name) == {:error, :model_not_found}
      assert MLServe.predict(name, 1) == {:error, :model_not_found}
      refute name in MLServe.models()
    end

    test "calls the backend's unload callback" do
      tok = token()
      name = load!(backend: Backends.Unloadable, config: [token: tok])

      :ok = MLServe.unload_model(name)

      eventually(fn -> Backends.count(tok, :unload) == 1 end)
    end

    test "stops the model's supervision subtree" do
      name = load!(backend: Backends.Pooled, workers: 2)
      pid = MLServe.ModelInstance.whereis(name, "1.0.0")
      ref = Process.monitor(pid)

      :ok = MLServe.unload_model(name)

      assert_receive {:DOWN, ^ref, :process, ^pid, _reason}, 1_000

      # Registry deregisters on a monitor message, so absence is eventual rather than immediate.
      assert eventually(fn -> MLServe.Worker.whereis(name, "1.0.0", 0) == nil end)
    end

    test "falls back to the newest surviving version when the default is unloaded" do
      name = load!(backend: Backends.Echo)
      {:ok, _} = MLServe.load_model(name, backend: Backends.Echo, version: "2.0.0")
      :ok = MLServe.await_ready({name, "2.0.0"})

      :ok = MLServe.unload_model(name, version: "1.0.0")

      assert MLServe.ModelRegistry.default_version(name) == {:ok, "2.0.0"}
      assert {:ok, 7} = MLServe.predict(name, 7)
    end

    test "erases shared state from persistent_term" do
      name = load!(backend: Backends.Echo)
      key = MLServe.Route.state_key(name, "1.0.0")
      assert :persistent_term.get(key, :missing) != :missing

      :ok = MLServe.unload_model(name)

      assert :persistent_term.get(key, :missing) == :missing
    end

    test "returns model_not_found for an unknown model" do
      assert MLServe.unload_model(:definitely_not_loaded) == {:error, :model_not_found}
    end

    test "drains in-flight requests before terminating workers" do
      name = load!(backend: Backends.Slow, workers: 1, config: [delay: 150], timeout: 2_000)

      task = Task.async(fn -> MLServe.predict(name, :slow) end)
      # Let the request reach the worker before unloading.
      eventually(fn -> match?({:ok, %{in_flight: 1}}, MLServe.model_status(name)) end)

      :ok = MLServe.unload_model(name, timeout: 1_000)

      assert Task.await(task, 2_000) == {:ok, :slow}
    end

    test "reports outstanding requests when the drain timeout expires" do
      name = load!(backend: Backends.Slow, workers: 1, config: [delay: 400], timeout: 2_000)
      ref = attach_telemetry([[:ml_serve, :model, :unload]])

      task = Task.async(fn -> MLServe.predict(name, :slow) end)
      eventually(fn -> match?({:ok, %{in_flight: 1}}, MLServe.model_status(name)) end)

      :ok = MLServe.unload_model(name, timeout: 50)

      assert {%{drained: 1}, _metadata} =
               assert_telemetry(ref, [:ml_serve, :model, :unload], name, 1_000)

      Task.shutdown(task, :brutal_kill)
    end
  end

  describe "reload_model/2" do
    test "reloads with the original configuration" do
      tok = token()
      name = load!(backend: Backends.Unloadable, config: [token: tok])

      assert {:ok, {^name, "1.0.0"}} = MLServe.reload_model(name)
      assert :ok = MLServe.await_ready(name)
      assert MLServe.predict(name, 1) == {:ok, 1}
      assert Backends.count(tok, :unload) == 1
    end

    test "accepts configuration overrides" do
      name = load!(backend: Backends.Pooled, workers: 2)

      assert {:ok, _} = MLServe.reload_model(name, workers: 5)
      assert :ok = MLServe.await_ready(name)
      assert {:ok, %{workers: 5}} = MLServe.model_status(name)
    end
  end

  describe "models/0 and versions/1" do
    test "lists registered models" do
      name = load!(backend: Backends.Echo)

      assert name in MLServe.models()
    end

    test "lists versions in ascending order, sorting numerically" do
      name = load!(backend: Backends.Echo, version: "9.0.0")

      for version <- ["10.0.0", "2.0.0"] do
        {:ok, _} = MLServe.load_model(name, backend: Backends.Echo, version: version)
      end

      :ok = MLServe.await_ready(name)

      assert {:ok, ["2.0.0", "9.0.0", "10.0.0"]} = MLServe.versions(name)
    end

    test "returns model_not_found for an unknown model" do
      assert MLServe.versions(:definitely_not_loaded) == {:error, :model_not_found}
    end
  end

  describe "ready?/1 and await_ready/2" do
    test "ready? is false while loading and true once loaded" do
      name = register!(backend: Backends.FailingLoad, config: [token: token()])
      refute MLServe.ready?(name)

      other = load!(backend: Backends.Echo)
      assert MLServe.ready?(other)
    end

    test "await_ready times out for a model that never loads" do
      name = register!(backend: Backends.FailingLoad, config: [token: token()])

      assert MLServe.await_ready(name, 50) == {:error, :timeout}
    end

    test "ready?/1 is false for an unknown model" do
      refute MLServe.ready?(:definitely_not_loaded)
    end
  end
end
