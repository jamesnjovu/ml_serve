defmodule MLServe.VersioningTest do
  use MLServe.Case, async: true

  alias MLServe.Test.Backends

  defp two_versions do
    name = load!(backend: MLServe.Backend.Static, config: [result: :v1])

    {:ok, _} =
      MLServe.load_model(name,
        backend: MLServe.Backend.Static,
        version: "2.0.0",
        config: [result: :v2]
      )

    :ok = MLServe.await_ready({name, "2.0.0"})
    name
  end

  describe "side-by-side versions" do
    test "both versions serve their own traffic" do
      name = two_versions()

      assert MLServe.predict(name, nil, version: "1.0.0") == {:ok, :v1}
      assert MLServe.predict(name, nil, version: "2.0.0") == {:ok, :v2}
    end

    test "unpinned traffic goes to the default version" do
      name = two_versions()

      assert MLServe.predict(name, nil) == {:ok, :v1}
    end

    test "each version reports its own status" do
      name = two_versions()

      assert {:ok, %{version: "1.0.0", default?: true}} = MLServe.model_status(name)

      assert {:ok, %{version: "2.0.0", default?: false}} =
               MLServe.model_status(name, version: "2.0.0")
    end

    test "each version has independent counters" do
      name = two_versions()

      for _ <- 1..5, do: MLServe.predict(name, nil, version: "1.0.0")
      for _ <- 1..2, do: MLServe.predict(name, nil, version: "2.0.0")

      assert {:ok, %{requests: 5}} = MLServe.model_status(name, version: "1.0.0")
      assert {:ok, %{requests: 2}} = MLServe.model_status(name, version: "2.0.0")
    end
  end

  describe "promote/2" do
    test "moves unpinned traffic to the new version" do
      name = two_versions()
      assert MLServe.predict(name, nil) == {:ok, :v1}

      assert :ok = MLServe.promote(name, "2.0.0")

      assert MLServe.predict(name, nil) == {:ok, :v2}
      assert {:ok, %{version: "2.0.0", default?: true}} = MLServe.model_status(name)
    end

    test "predictions keep succeeding across the switch" do
      name = two_versions()

      task =
        Task.async(fn ->
          for _ <- 1..200, do: MLServe.predict(name, nil)
        end)

      Process.sleep(5)
      :ok = MLServe.promote(name, "2.0.0")

      results = Task.await(task, 5_000)

      # Not one request fails while the default pointer flips.
      assert Enum.all?(results, &match?({:ok, _}, &1))
      assert Enum.any?(results, &(&1 == {:ok, :v1}))
    end

    test "returns model_not_found for an unknown version" do
      name = two_versions()

      assert MLServe.promote(name, "9.9.9") == {:error, :model_not_found}
    end
  end

  describe "canary/3" do
    test "splits unpinned traffic between the default and the candidate" do
      name = two_versions()
      assert :ok = MLServe.canary(name, "2.0.0", 50)

      counts =
        for _ <- 1..1_000, do: MLServe.predict(name, nil)

      frequencies = Enum.frequencies(counts)

      assert Map.has_key?(frequencies, {:ok, :v1})
      assert Map.has_key?(frequencies, {:ok, :v2})
      # Wide tolerance: this asserts the split happens, not that :rand is well-behaved.
      assert_in_delta frequencies[{:ok, :v2}], 500, 120
    end

    test "a small weight sends most traffic to the default" do
      name = two_versions()
      :ok = MLServe.canary(name, "2.0.0", 5)

      frequencies = for(_ <- 1..1_000, do: MLServe.predict(name, nil)) |> Enum.frequencies()

      assert frequencies[{:ok, :v1}] > 850
    end

    test "pinned requests ignore the canary entirely" do
      name = two_versions()
      :ok = MLServe.canary(name, "2.0.0", 100)

      results = for _ <- 1..50, do: MLServe.predict(name, nil, version: "1.0.0")

      assert Enum.all?(results, &(&1 == {:ok, :v1}))
    end

    test "telemetry tags the serving version and marks canary traffic" do
      ref = attach_telemetry([[:ml_serve, :prediction, :stop]])
      name = two_versions()
      :ok = MLServe.canary(name, "2.0.0", 100)

      MLServe.predict(name, nil)

      {_measurements, metadata} =
        assert_telemetry(ref, [:ml_serve, :prediction, :stop], name)

      assert metadata.version == "2.0.0"
      assert metadata.canary?
    end

    test "default traffic is not marked as canary" do
      ref = attach_telemetry([[:ml_serve, :prediction, :stop]])
      name = two_versions()
      :ok = MLServe.canary(name, "2.0.0", 1)

      MLServe.predict(name, nil, version: "1.0.0")

      {_measurements, metadata} =
        assert_telemetry(ref, [:ml_serve, :prediction, :stop], name)

      refute metadata.canary?
    end

    test "traffic falls back to the default when the candidate is not ready" do
      name = load!(backend: MLServe.Backend.Static, config: [result: :v1])

      {:ok, _} =
        MLServe.load_model(name,
          backend: Backends.FailingLoad,
          version: "2.0.0",
          config: [token: token()]
        )

      :ok = MLServe.canary(name, "2.0.0", 100)

      # A candidate that cannot serve must not black-hole its share of traffic.
      results = for _ <- 1..50, do: MLServe.predict(name, nil)
      assert Enum.all?(results, &(&1 == {:ok, :v1}))
    end

    test "promote clears the canary" do
      name = two_versions()
      :ok = MLServe.canary(name, "2.0.0", 50)
      assert {:ok, %{canary: {"2.0.0", 50}}} = MLServe.model_status(name)

      :ok = MLServe.promote(name, "2.0.0")

      assert {:ok, %{canary: nil}} = MLServe.model_status(name, version: "2.0.0")
      assert Enum.all?(for(_ <- 1..50, do: MLServe.predict(name, nil)), &(&1 == {:ok, :v2}))
    end

    test "clear_canary/1 aborts a rollout" do
      name = two_versions()
      :ok = MLServe.canary(name, "2.0.0", 100)

      :ok = MLServe.clear_canary(name)

      assert Enum.all?(for(_ <- 1..50, do: MLServe.predict(name, nil)), &(&1 == {:ok, :v1}))
    end

    test "refuses to canary the version that is already the default" do
      name = two_versions()

      assert MLServe.canary(name, "1.0.0", 10) == {:error, :already_default}
    end

    test "refuses an unknown version" do
      name = two_versions()

      assert MLServe.canary(name, "9.9.9", 10) == {:error, :model_not_found}
    end

    test "unloading the canary version clears the canary" do
      name = two_versions()
      :ok = MLServe.canary(name, "2.0.0", 50)

      :ok = MLServe.unload_model(name, version: "2.0.0")

      assert MLServe.ModelRegistry.canary(name) == nil
      assert Enum.all?(for(_ <- 1..20, do: MLServe.predict(name, nil)), &(&1 == {:ok, :v1}))
    end
  end

  describe "the full rollout" do
    test "load, canary, promote, drain — with no failed request throughout" do
      name = load!(backend: MLServe.Backend.Static, config: [result: :v1])

      collector =
        Task.async(fn ->
          for _ <- 1..600 do
            result = MLServe.predict(name, nil)
            Process.sleep(1)
            result
          end
        end)

      # 1. Load the candidate beside the live version.
      {:ok, _} =
        MLServe.load_model(name,
          backend: MLServe.Backend.Static,
          version: "2.0.0",
          config: [result: :v2]
        )

      :ok = MLServe.await_ready({name, "2.0.0"})

      # 2. Send it a slice of traffic and confirm both versions are being exercised.
      :ok = MLServe.canary(name, "2.0.0", 50)
      Process.sleep(50)

      # 3. Promote once the candidate looks healthy.
      :ok = MLServe.promote(name, "2.0.0")

      # 4. Drain and remove the old version.
      :ok = MLServe.unload_model(name, version: "1.0.0")

      results = Task.await(collector, 10_000)

      assert length(results) == 600
      assert Enum.all?(results, &match?({:ok, _}, &1)), "a request failed during the rollout"
      assert Enum.any?(results, &(&1 == {:ok, :v1}))
      assert Enum.any?(results, &(&1 == {:ok, :v2}))
      assert List.last(results) == {:ok, :v2}
      assert {:ok, ["2.0.0"]} = MLServe.versions(name)
    end
  end
end
