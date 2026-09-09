defmodule MLServe.BackendTest do
  use MLServe.Case, async: true

  doctest MLServe.Backend
  doctest MLServe.Backend.Function
  doctest MLServe.Backend.Static
  doctest MLServe.ModelSpec
  doctest MLServe.Cache, only: [key: 3]
  doctest MLServe.Telemetry

  alias MLServe.Test.Backends

  describe "MLServe.Backend.Function" do
    test "wraps a plain function" do
      name = load!(backend: MLServe.Backend.Function, config: [predict: &(&1 * 2)])

      assert MLServe.predict(name, 21) == {:ok, 42}
    end

    test "accepts an MFA tuple" do
      name =
        load!(backend: MLServe.Backend.Function, config: [predict: {__MODULE__, :triple, []}])

      assert MLServe.predict(name, 3) == {:ok, 9}
    end

    test "passes through ok and error tuples unchanged" do
      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [predict: fn n -> if n > 0, do: {:ok, n}, else: {:error, :negative} end]
        )

      assert MLServe.predict(name, 1) == {:ok, 1}
      assert MLServe.predict(name, -1) == {:error, :negative}
    end

    test "an :init function supplies state to every prediction" do
      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [
            init: fn -> %{"a" => 1, "b" => 2} end,
            predict: fn {table, key} -> {:ok, Map.get(table, key)} end
          ]
        )

      assert MLServe.predict(name, "a") == {:ok, 1}
      assert MLServe.predict(name, "z") == {:ok, nil}
    end

    test "requires a predict function" do
      name = register!(backend: MLServe.Backend.Function, config: [])

      eventually(fn -> match?({:ok, %{status: :failed}}, MLServe.model_status(name)) end, 3_000)

      assert {:ok, %{failure: message}} = MLServe.model_status(name)
      assert message =~ "requires a :predict function"
    end

    test "runs shared, in the calling process" do
      name =
        load!(backend: MLServe.Backend.Function, config: [predict: fn _ -> {:ok, self()} end])

      assert {:ok, pid} = MLServe.predict(name, nil)
      assert pid == self()
    end

    test "reports whether it batches natively" do
      plain = load!(backend: MLServe.Backend.Function, config: [predict: & &1])
      assert {:ok, %{metadata: %{native_batching: false}}} = MLServe.model_status(plain)

      batching =
        load!(
          backend: MLServe.Backend.Function,
          config: [predict: & &1, batch_predict: &Enum.reverse/1]
        )

      assert {:ok, %{metadata: %{native_batching: true}}} = MLServe.model_status(batching)
    end
  end

  describe "MLServe.Backend.Static" do
    test "returns the configured result for any input" do
      name = load!(backend: MLServe.Backend.Static, config: [result: %{prediction: :fraud}])

      assert MLServe.predict(name, :anything) == {:ok, %{prediction: :fraud}}
      assert MLServe.predict(name, %{totally: :different}) == {:ok, %{prediction: :fraud}}
    end

    test "returns the configured error" do
      name = load!(backend: MLServe.Backend.Static, config: [error: :service_unavailable])

      assert MLServe.predict(name, :x) == {:error, :service_unavailable}
    end

    test "batches by duplicating the result" do
      name = load!(backend: MLServe.Backend.Static, config: [result: :same])

      assert MLServe.batch_predict(name, [1, 2, 3]) == {:ok, [:same, :same, :same]}
    end

    test "a delay makes timeouts testable" do
      # Forced to :exclusive: Static declares :shared, and a shared backend runs in the caller,
      # where there is no other process to abandon and :timeout cannot be enforced.
      name =
        load!(
          backend: MLServe.Backend.Static,
          concurrency: :exclusive,
          workers: 1,
          config: [result: :slow, delay: 100]
        )

      assert MLServe.predict(name, :x, timeout: 10) == {:error, :timeout}
    end

    test "a shared backend ignores :timeout because it runs in the caller" do
      name = load!(backend: MLServe.Backend.Static, config: [result: :slow, delay: 30])

      # Documented behaviour, asserted so it cannot regress silently.
      assert MLServe.predict(name, :x, timeout: 1) == {:ok, :slow}
    end
  end

  describe "capability detection" do
    test "supports_batching?/1 reflects the exported callback" do
      assert MLServe.Backend.supports_batching?(Backends.Counting)
      refute MLServe.Backend.supports_batching?(Backends.Echo)
      refute MLServe.Backend.supports_batching?(NoSuchModule)
    end

    test "a backend with no capabilities/0 defaults to exclusive and :once" do
      name = load!(backend: Backends.Minimal, workers: 2)

      assert {:ok, %{concurrency: :exclusive, workers: 2}} = MLServe.model_status(name)
      assert MLServe.predict(name, :x) == {:ok, :x}
    end

    test "optional callbacks are genuinely optional" do
      # Minimal implements only load/1 and predict/2 — no metadata/1, unload/1 or batch_predict/2.
      name = load!(backend: Backends.Minimal, workers: 1)

      assert {:ok, %{metadata: %{}}} = MLServe.model_status(name)
      assert MLServe.batch_predict(name, [1, 2]) == {:ok, [1, 2]}
      assert :ok = MLServe.unload_model(name)
    end
  end

  @doc false
  def triple(n), do: {:ok, n * 3}
end
