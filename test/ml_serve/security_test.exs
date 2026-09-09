defmodule MLServe.SecurityTest do
  use MLServe.Case, async: true

  alias MLServe.Security
  alias MLServe.Test.Backends

  setup do
    root = Path.join(System.tmp_dir!(), "ml_serve_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(Path.join(root, "nested"))
    model = Path.join(root, "model.bin")
    File.write!(model, "weights")

    on_exit(fn -> File.rm_rf(root) end)

    {:ok, root: root, model: model}
  end

  describe "validate_path/2" do
    test "accepts a file inside the root", %{root: root} do
      assert {:ok, resolved} = Security.validate_path("model.bin", root: root)
      assert resolved == Path.join(Path.expand(root), "model.bin")
    end

    test "accepts an absolute path inside the root", %{root: root, model: model} do
      assert {:ok, _resolved} = Security.validate_path(model, root: root)
    end

    # Containment is decided on the expanded path, so the result does not depend on whether the
    # traversal target happens to exist — `System.tmp_dir!()` is `/tmp` on Linux but
    # `/var/folders/...` on macOS, and this asserted :enoent on the latter.
    test "rejects traversal out of the root", %{root: root} do
      assert Security.validate_path("../../etc/passwd", root: root) ==
               {:error, {:invalid_path, :outside_root}}
    end

    test "rejects traversal out of the root even when the target does not exist", %{root: root} do
      assert Security.validate_path("../../nowhere/at/all.bin", root: root) ==
               {:error, {:invalid_path, :outside_root}}
    end

    test "rejects an absolute path outside the root", %{root: root} do
      assert Security.validate_path("/etc/hosts", root: root) ==
               {:error, {:invalid_path, :outside_root}}
    end

    test "rejects a symlink that escapes the root", %{root: root} do
      outside =
        Path.join(System.tmp_dir!(), "ml_serve_outside_#{System.unique_integer([:positive])}")

      File.write!(outside, "secret")
      link = Path.join(root, "escape.bin")
      :ok = File.ln_s(outside, link)
      on_exit(fn -> File.rm_rf(outside) end)

      # Symlinks are resolved *before* the containment check; otherwise a link inside the root
      # pointing at /etc would sail straight through.
      assert Security.validate_path("escape.bin", root: root) ==
               {:error, {:invalid_path, :outside_root}}
    end

    test "rejects a missing file", %{root: root} do
      assert Security.validate_path("nope.bin", root: root) == {:error, {:invalid_path, :enoent}}
    end

    test "rejects a directory", %{root: root} do
      assert Security.validate_path("nested", root: root) ==
               {:error, {:invalid_path, :not_a_regular_file}}
    end

    test "rejects a file over the size limit", %{root: root} do
      assert Security.validate_path("model.bin", root: root, max_bytes: 2) ==
               {:error, {:invalid_path, :too_large}}
    end

    test "validate_path!/2 raises with an actionable message", %{root: root} do
      assert_raise MLServe.Error, ~r/resolves outside the configured :model_root/, fn ->
        Security.validate_path!("/etc/hosts", root: root)
      end
    end
  end

  describe "checksums" do
    test "accepts a matching digest", %{root: root, model: model} do
      {:ok, digest} = Security.digest(model, :sha256)

      assert {:ok, _} =
               Security.validate_path("model.bin", root: root, checksum: {:sha256, digest})
    end

    test "accepts an uppercase digest", %{root: root, model: model} do
      {:ok, digest} = Security.digest(model, :sha256)
      upper = String.upcase(digest)

      assert {:ok, _} =
               Security.validate_path("model.bin", root: root, checksum: {:sha256, upper})
    end

    test "rejects a mismatched digest", %{root: root} do
      assert {:error, {:checksum_mismatch, actual}} =
               Security.validate_path("model.bin",
                 root: root,
                 checksum: {:sha256, String.duplicate("0", 64)}
               )

      assert String.length(actual) == 64
    end

    test "supports sha512", %{root: root, model: model} do
      {:ok, digest} = Security.digest(model, :sha512)

      assert {:ok, _} =
               Security.validate_path("model.bin", root: root, checksum: {:sha512, digest})
    end

    test "a nil checksum skips verification" do
      assert Security.verify_checksum("/nonexistent/path", nil) == :ok
    end
  end

  describe "validate_backend/1" do
    test "accepts a module implementing MLServe.Model" do
      assert Security.validate_backend(Backends.Echo) == :ok
    end

    test "rejects a module that is not loaded" do
      assert Security.validate_backend(NoSuchModuleAtAll) ==
               {:error, {:invalid_backend, :not_loaded}}
    end

    test "rejects a loaded module lacking the callbacks" do
      assert Security.validate_backend(Enum) == {:error, {:invalid_backend, :not_a_model}}
    end

    test "rejects a non-module" do
      assert Security.validate_backend("Elixir.Enum") == {:error, {:invalid_backend, :not_loaded}}
    end
  end

  describe "model loading integration" do
    test "a model with a bad path fails to load rather than reaching the backend", %{root: root} do
      name =
        register!(
          backend: Backends.Echo,
          path: "../escape.bin",
          config: [root: root]
        )

      eventually(fn -> match?({:ok, %{status: :failed}}, MLServe.model_status(name)) end, 3_000)

      assert {:ok, %{failure: {:invalid_path, _}}} = MLServe.model_status(name)
    end
  end
end
