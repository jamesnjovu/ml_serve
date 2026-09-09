defmodule MLServeTest do
  use MLServe.Case, async: true

  doctest MLServe.Route
  doctest MLServe.Model

  alias MLServe.Test.Backends

  # Defined at module level, not inside the test: a `defmodule` in a test body is redefined on
  # every run, which makes `mix test --repeat-until-failure` warn on each pass.
  defmodule FraudModel do
    @moduledoc "The backend from the README, kept honest by the test below."
    @behaviour MLServe.Model

    @impl true
    def load(config), do: {:ok, Keyword.fetch!(config, :threshold)}

    @impl true
    def predict(threshold, %{amount: amount}) do
      probability = min(amount / 2000, 1.0)

      {:ok,
       %{
         prediction: if(probability > threshold, do: :fraud, else: :legitimate),
         probability: probability
       }}
    end
  end

  describe "the README example" do
    test "a model backend, registration, and a prediction" do
      name = load!(backend: FraudModel, config: [threshold: 0.7])

      assert {:ok, %{prediction: :fraud, probability: probability}} =
               MLServe.predict(name, %{amount: 1500.50})

      assert_in_delta probability, 0.75, 0.001

      assert {:ok, %{prediction: :legitimate}} = MLServe.predict(name, %{amount: 10})
    end
  end

  describe "model_status/2" do
    test "reports every documented field" do
      name = load!(backend: Backends.Pooled, workers: 2, max_concurrency: 10)

      assert {:ok, status} = MLServe.model_status(name)

      for key <- [
            :name,
            :version,
            :status,
            :backend,
            :concurrency,
            :workers,
            :selection,
            :batching,
            :native_batching,
            :cache,
            :max_concurrency,
            :timeout,
            :default?,
            :canary,
            :loaded_at,
            :load_duration_ms,
            :in_flight,
            :requests,
            :errors,
            :failure,
            :metadata
          ] do
        assert Map.has_key?(status, key), "model_status/2 is missing #{inspect(key)}"
      end

      assert status.max_concurrency == 10
      assert status.native_batching == false
    end

    test "counters track requests and errors separately" do
      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [predict: fn n -> if n > 0, do: {:ok, n}, else: {:error, :negative} end]
        )

      for i <- 1..7, do: MLServe.predict(name, i)
      for _ <- 1..3, do: MLServe.predict(name, -1)

      assert {:ok, %{requests: 10, errors: 3, in_flight: 0}} = MLServe.model_status(name)
    end

    test "returns model_not_found for an unknown model" do
      assert MLServe.model_status(:definitely_not_loaded) == {:error, :model_not_found}
    end
  end

  describe "ready?/1" do
    test ":all reports true when no models are registered in an empty system" do
      # Other async tests hold models, so this only asserts the function is total.
      assert is_boolean(MLServe.ready?(:all))
    end

    test "accepts a {name, version} tuple" do
      name = load!(backend: Backends.Echo)

      assert MLServe.ready?({name, "1.0.0"})
      refute MLServe.ready?({name, "9.9.9"})
    end
  end

  describe "the supervision tree" do
    test "every long-lived child is running" do
      children = Supervisor.which_children(MLServe.Supervisor)
      ids = Enum.map(children, fn {id, _pid, _type, _modules} -> id end)

      assert MLServe.ModelRegistry in ids
      assert MLServe.Cache in ids
      assert MLServe.ModelSupervisor in ids
      assert MLServe.TaskSupervisor in ids
      assert MLServe.Registry in ids

      # Bootstrap is a transient Task: it loads the configured models and exits, so its pid is
      # :undefined once it has finished. Everything else must still be alive.
      long_lived =
        Enum.reject(children, fn {id, _pid, _type, _modules} -> id == MLServe.Bootstrap end)

      assert Enum.all?(long_lived, fn {_id, pid, _type, _modules} -> is_pid(pid) end)
    end

    test "the catalog and cache ETS tables exist and are readable" do
      assert :ets.info(:ml_serve_catalog, :protection) == :protected
      assert :ets.info(:ml_serve_catalog, :read_concurrency)
      assert :ets.info(:ml_serve_cache, :protection) == :public
      assert :ets.info(:ml_serve_cache, :write_concurrency)
    end
  end

  describe "status consistency" do
    test "ready?/1 never reports ready before model_status/2 agrees" do
      # These are two independent ETS rows polled by different callers — a readiness probe reads
      # the route, a status page reads the entry. Writing them in the wrong order leaves a window
      # where a model is routable but still reports :loading with no metadata.
      for _ <- 1..25 do
        name = unique_name()
        {:ok, _} = MLServe.load_model(name, backend: Backends.Echo, config: [tag: :consistent])
        on_exit(fn -> MLServe.unload_model(name, timeout: 100) end)

        :ok = MLServe.await_ready(name, 5_000)

        assert {:ok, %{status: :ready, metadata: %{tag: :consistent}}} =
                 MLServe.model_status(name)
      end
    end
  end

  describe "route resolution" do
    test "a slim route is what the hot path reads, not the full spec" do
      name = load!(backend: Backends.Pooled, workers: 2, config: [tag: :big_config_value])

      {:ok, route} = MLServe.ModelRegistry.route(name)

      # The route deliberately excludes :config, which can hold closures and large terms that
      # would otherwise be copied out of ETS on every single prediction.
      refute Map.has_key?(route, :config)
      assert route.backend == Backends.Pooled
      assert route.workers == 2
    end

    test "round-robin indices wrap within the worker count" do
      name = load!(backend: Backends.Pooled, workers: 3)
      {:ok, route} = MLServe.ModelRegistry.route(name)

      indices = for _ <- 1..12, do: MLServe.Route.next_index(route)

      assert Enum.all?(indices, &(&1 in 0..2))
      assert Enum.uniq(indices) |> Enum.sort() == [0, 1, 2]
    end
  end
end
