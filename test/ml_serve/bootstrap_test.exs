defmodule MLServe.BootstrapTest do
  use MLServe.Case, async: true

  alias MLServe.Bootstrap
  alias MLServe.Test.Backends

  describe "run/1" do
    test "loads models passed explicitly" do
      name = unique_name()
      on_exit(fn -> MLServe.unload_model(name, timeout: 100) end)

      assert :ok = Bootstrap.run(models: [{name, [backend: Backends.Echo]}], start_mode: :async)
      assert :ok = MLServe.await_ready(name, 2_000)
      assert MLServe.predict(name, :x) == {:ok, :x}
    end

    test "start_mode: :sync waits for models to become ready" do
      name = unique_name()
      on_exit(fn -> MLServe.unload_model(name, timeout: 100) end)

      assert :ok = Bootstrap.run(models: [{name, [backend: Backends.Echo]}], start_mode: :sync)

      # No await_ready here: :sync means run/1 does not return until the model can serve.
      assert MLServe.predict(name, :x) == {:ok, :x}
    end

    test "loads several models" do
      names = for _ <- 1..3, do: unique_name()
      on_exit(fn -> Enum.each(names, &MLServe.unload_model(&1, timeout: 100)) end)

      models = Enum.map(names, &{&1, [backend: Backends.Echo]})

      assert :ok = Bootstrap.run(models: models, start_mode: :sync)
      assert Enum.all?(names, &MLServe.ready?/1)
    end

    test "a misconfigured model is logged and skipped, not fatal" do
      good = unique_name()
      bad = unique_name()
      on_exit(fn -> MLServe.unload_model(good, timeout: 100) end)

      models = [
        # Enum does not implement MLServe.Model, so this one fails validation outright.
        {bad, [backend: Enum]},
        {good, [backend: Backends.Echo]}
      ]

      assert :ok = Bootstrap.run(models: models, start_mode: :sync)

      # One bad entry must not stop the rest of the fleet from loading.
      assert MLServe.ready?(good)
      refute bad in MLServe.models()
    end

    test "an empty model list is a no-op" do
      assert :ok = Bootstrap.run(models: [], start_mode: :sync)
    end

    test "a model that fails to load does not block :sync startup forever" do
      name = unique_name()
      on_exit(fn -> MLServe.unload_model(name, timeout: 100) end)

      models = [{name, [backend: Backends.FailingLoad, config: [token: token()]]}]

      # await_ready/2 is bounded, so a permanently broken model delays boot rather than hanging it.
      assert :ok = Bootstrap.run(models: models, start_mode: :async)
      eventually(fn -> match?({:ok, %{status: :failed}}, MLServe.model_status(name)) end, 3_000)
    end
  end
end
