defmodule MLServe.InternalsTest do
  @moduledoc """
  Covers internal paths that only surface under failure or shutdown — the ones a happy-path suite
  never reaches, and which therefore break silently.
  """
  use MLServe.Case, async: true

  alias MLServe.ModelServer
  alias MLServe.Security
  alias MLServe.Test.Backends

  describe "MLServe.Security edge cases" do
    setup do
      root = Path.join(System.tmp_dir!(), "ml_serve_int_#{System.unique_integer([:positive])}")
      File.mkdir_p!(root)
      model = Path.join(root, "model.bin")
      File.write!(model, "weights")
      on_exit(fn -> File.rm_rf(root) end)

      {:ok, root: root, model: model}
    end

    test "digest/2 reports a missing file rather than raising" do
      assert {:error, {:invalid_path, :enoent}} =
               Security.digest("/definitely/not/here.bin", :sha256)
    end

    test "digest/2 streams large files", %{root: root} do
      # Larger than the 2MB read chunk, so the streaming reduce is actually exercised.
      big = Path.join(root, "big.bin")
      File.write!(big, :binary.copy(<<0>>, 5 * 1024 * 1024))

      assert {:ok, digest} = Security.digest(big, :sha256)
      assert String.length(digest) == 64
      assert digest == :sha256 |> :crypto.hash(File.read!(big)) |> Base.encode16(case: :lower)
    end

    test "validate_path!/2 returns the resolved path on success", %{root: root} do
      assert Security.validate_path!("model.bin", root: root) ==
               Path.join(Path.expand(root), "model.bin")
    end

    test "a checksum of the wrong length is rejected, not compared", %{root: root} do
      # Guards the constant-time comparison: :crypto.hash_equals/2 raises on differing sizes.
      assert {:error, {:checksum_mismatch, _}} =
               Security.validate_path("model.bin", root: root, checksum: {:sha256, "abc"})
    end

    test "an sha512 mismatch is reported", %{root: root} do
      assert {:error, {:checksum_mismatch, actual}} =
               Security.validate_path("model.bin",
                 root: root,
                 checksum: {:sha512, String.duplicate("0", 128)}
               )

      assert String.length(actual) == 128
    end

    test "a checksum failure surfaces through model loading" do
      {path, _digest} = model_fixture!()

      name =
        register!(
          backend: Backends.Echo,
          path: path,
          checksum: {:sha256, String.duplicate("a", 64)}
        )

      eventually(fn -> match?({:ok, %{status: :failed}}, MLServe.model_status(name)) end, 3_000)

      assert {:ok, %{failure: {:checksum_mismatch, _}}} = MLServe.model_status(name)
    end

    test "a valid checksum lets the model load" do
      {path, digest} = model_fixture!()

      name = load!(backend: Backends.Echo, path: path, checksum: {:sha256, digest})

      assert MLServe.predict(name, :x) == {:ok, :x}
    end

    test "the resolved path is passed to the backend" do
      {path, _digest} = model_fixture!()

      name =
        load!(
          backend: MLServe.Backend.Function,
          path: path,
          config: [predict: fn _ -> {:ok, :ok} end]
        )

      assert {:ok, spec} = MLServe.ModelRegistry.spec(name, "1.0.0")
      assert spec.path == Path.expand(path)
    end
  end

  describe "MLServe.ModelServer" do
    test "state/2 returns the loaded backend state" do
      name = load!(backend: Backends.Echo, config: [tag: :tagged])

      assert {:ok, :tagged} = ModelServer.state(name, "1.0.0")
    end

    test "state/2 reports an unknown model" do
      assert ModelServer.state(:definitely_not_loaded, "1.0.0") == {:error, :model_not_ready}
    end

    test "drain/3 reports an unknown model" do
      assert ModelServer.drain(:definitely_not_loaded, "1.0.0", 10) == {:error, :model_not_found}
    end

    test "unexpected messages do not crash the server" do
      name = load!(backend: Backends.Echo)
      pid = ModelServer.whereis(name, "1.0.0")

      send(pid, :something_unexpected)

      assert :sys.get_state(pid)
      assert Process.alive?(pid)
      assert MLServe.predict(name, :x) == {:ok, :x}
    end

    test "per-worker state is released when the pool shuts down" do
      tok = token()
      name = load!(backend: Backends.PerWorker, workers: 3, config: [token: tok])

      assert Backends.count(tok, :load) == 3

      :ok = MLServe.unload_model(name)

      # Each worker owns its own resource, so each must release it.
      eventually(fn -> Backends.count(tok, :unload) == 3 end)
    end
  end

  describe "MLServe.Backend.Function" do
    test "an MFA batch_predict is used for batches" do
      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [predict: &(&1 * 2), batch_predict: {__MODULE__, :triple_all, []}]
        )

      assert MLServe.batch_predict(name, [1, 2, 3]) == {:ok, [3, 6, 9]}
    end

    test "a stateful batch_predict receives {state, inputs}" do
      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [
            init: fn -> 10 end,
            predict: fn {factor, n} -> {:ok, n * factor} end,
            batch_predict: fn {factor, ns} -> Enum.map(ns, &(&1 * factor)) end
          ]
        )

      assert MLServe.predict(name, 3) == {:ok, 30}
      assert MLServe.batch_predict(name, [1, 2]) == {:ok, [10, 20]}
    end

    test "batch_predict returning a non-list is reported" do
      name =
        load!(
          backend: MLServe.Backend.Function,
          config: [predict: & &1, batch_predict: fn _ -> :not_a_list end]
        )

      assert {:error, {:invalid_return, :not_a_list}} = MLServe.batch_predict(name, [1, 2])
    end

    test "a malformed predict option is rejected at load" do
      name = register!(backend: MLServe.Backend.Function, config: [predict: :nope])

      eventually(fn -> match?({:ok, %{status: :failed}}, MLServe.model_status(name)) end, 3_000)

      assert {:ok, %{failure: message}} = MLServe.model_status(name)
      assert message =~ "must be a 1-arity function"
    end
  end

  describe "dispatch fallbacks" do
    test "a shared model whose state is gone reports not ready" do
      name = load!(backend: Backends.Echo)
      :persistent_term.erase(MLServe.Route.state_key(name, "1.0.0"))

      assert MLServe.predict(name, :x) == {:error, :model_not_ready}
      assert MLServe.batch_predict(name, [1]) == {:error, :model_not_ready}
    end

    test "least_loaded picks between two live workers" do
      name = load!(backend: Backends.Pooled, workers: 2, selection: :least_loaded)
      {:ok, route} = MLServe.ModelRegistry.route(name)

      assert MLServe.Dispatcher.select_worker(route) != nil
    end

    test "select_worker returns nil for a model with no workers" do
      name = load!(backend: Backends.Echo)
      {:ok, route} = MLServe.ModelRegistry.route(name)

      assert MLServe.Dispatcher.select_worker(route) == nil
    end

    test "an infinite timeout is accepted" do
      name = load!(backend: Backends.Pooled, workers: 1)

      assert {:ok, {:x, _pid}} = MLServe.predict(name, :x, timeout: :infinity)
    end
  end

  @doc false
  def triple_all(ns), do: Enum.map(ns, &(&1 * 3))
end
