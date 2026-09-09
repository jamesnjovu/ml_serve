defmodule MLServe.ConfigTest do
  use MLServe.Case, async: true

  doctest MLServe.Config

  alias MLServe.Config
  alias MLServe.ModelSpec
  alias MLServe.Test.Backends

  describe "build/2 defaults" do
    test "fills in sensible defaults" do
      assert {:ok, spec} = Config.build(:m, backend: Backends.Echo)

      assert spec.name == :m
      assert spec.version == "1.0.0"
      assert spec.workers == System.schedulers_online()
      assert spec.timeout == 5_000
      assert spec.max_concurrency == :infinity
      assert spec.selection == :round_robin
      assert spec.batching == nil
      assert spec.cache == nil
      refute spec.restart_on_error
    end

    test "injects the version into the backend config" do
      {:ok, spec} = Config.build(:m, backend: Backends.Echo, version: "3.1.4")

      assert Keyword.get(spec.config, :version) == "3.1.4"
    end

    test "takes concurrency and load mode from the backend's capabilities" do
      {:ok, shared} = Config.build(:m, backend: Backends.Echo)
      assert shared.concurrency == :shared
      assert ModelSpec.worker_count(shared) == 0

      {:ok, exclusive} = Config.build(:m, backend: Backends.Pooled, workers: 3)
      assert exclusive.concurrency == :exclusive
      assert ModelSpec.worker_count(exclusive) == 3

      {:ok, per_worker} = Config.build(:m, backend: Backends.PerWorker)
      assert per_worker.load == :per_worker
    end

    test "explicit options override the backend's declared capabilities" do
      {:ok, spec} = Config.build(:m, backend: Backends.Echo, concurrency: :exclusive, workers: 2)

      assert spec.concurrency == :exclusive
      assert ModelSpec.worker_count(spec) == 2
    end
  end

  describe "build/2 validation" do
    test "requires a backend" do
      assert {:error, error} = Config.build(:m, [])
      assert error.type == :config
      assert Exception.message(error) =~ ":backend option is required"
    end

    test "rejects a module that is not loaded" do
      assert {:error, error} = Config.build(:m, backend: NoSuchModuleAnywhere)
      assert Exception.message(error) =~ "is not a loaded module"
    end

    test "rejects a module that does not implement MLServe.Model" do
      assert {:error, error} = Config.build(:m, backend: Enum)
      assert Exception.message(error) =~ "does not implement MLServe.Model"
      assert Exception.message(error) =~ "load/1 and predict/2 are required"
    end

    test "rejects unknown options and lists the valid ones" do
      assert {:error, error} = Config.build(:m, backend: Backends.Echo, worker: 4)
      assert Exception.message(error) =~ "unknown option :worker"
      assert Exception.message(error) =~ ":workers"
    end

    test "rejects an invalid worker count" do
      assert {:error, error} = Config.build(:m, backend: Backends.Pooled, workers: 0)
      assert Exception.message(error) =~ ":workers must be a positive integer"
    end

    test "rejects an empty version" do
      assert {:error, error} = Config.build(:m, backend: Backends.Echo, version: "")
      assert Exception.message(error) =~ ":version must be a non-empty string"
    end

    test "rejects an invalid selection strategy" do
      assert {:error, error} = Config.build(:m, backend: Backends.Pooled, selection: :fastest)
      assert Exception.message(error) =~ ":selection must be"
    end

    test "rejects an invalid max_concurrency" do
      assert {:error, error} = Config.build(:m, backend: Backends.Echo, max_concurrency: -1)

      assert Exception.message(error) =~
               ":max_concurrency must be a positive integer or :infinity"
    end

    test "rejects a malformed hook" do
      assert {:error, error} =
               Config.build(:m, backend: Backends.Echo, preprocess: :not_a_function)

      assert Exception.message(error) =~ "must be a {module, function, args} tuple"
    end

    test "rejects load: :per_worker with concurrency: :shared" do
      assert {:error, error} =
               Config.build(:m, backend: Backends.Echo, concurrency: :shared, load: :per_worker)

      assert Exception.message(error) =~ "meaningless with concurrency: :shared"
    end

    test "names the offending model in every message" do
      assert {:error, error} = Config.build(:fraud_detection, [])
      assert Exception.message(error) =~ "model :fraud_detection"
      assert error.details.model == :fraud_detection
    end

    test "build!/2 raises" do
      assert_raise MLServe.Error, ~r/:backend option is required/, fn ->
        Config.build!(:m, [])
      end
    end
  end

  describe "build/2 normalisation" do
    test "batching options are normalised into a map with defaults" do
      {:ok, spec} = Config.build(:m, backend: Backends.Echo, batching: [max_size: 32])

      assert spec.batching == %{max_size: 32, timeout: 10}
    end

    test "batching: false disables batching" do
      {:ok, spec} = Config.build(:m, backend: Backends.Echo, batching: false)

      assert spec.batching == nil
    end

    test "cache: true is expanded to the default TTL" do
      {:ok, spec} = Config.build(:m, backend: Backends.Echo, cache: true)

      assert spec.cache.enabled
      assert is_integer(spec.cache.ttl)
    end
  end

  describe "load_model/2 surfaces configuration errors" do
    test "returns the config error rather than starting anything" do
      assert {:error, error} = MLServe.load_model(unique_name(), backend: Enum)
      assert error.type == :config
    end
  end

  describe "application configuration accessors" do
    test "expose defaults" do
      assert is_integer(Config.default_timeout())
      assert is_integer(Config.max_batch_size())
      assert Config.start_mode() in [:async, :sync]
      assert is_binary(Config.model_root())

      cache = Config.cache()
      assert is_boolean(cache.enabled)
      assert is_integer(cache.ttl)
      assert is_integer(cache.max_size)
    end
  end
end
