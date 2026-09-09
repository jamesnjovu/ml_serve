defmodule MLServe.Security do
  @moduledoc """
  Model artifact validation: path containment, size limits, and integrity checks.

  MLServe treats a model file as **data, never as code**. The core library never calls
  `:erlang.binary_to_term/1`, `Code.eval_string/1`, or loads a NIF from a path supplied through
  configuration. Supplying a model file must not be a way to execute arbitrary code, so this
  module validates the path before any backend sees it.

  > #### Backend authors {: .warning}
  >
  > These guarantees stop at MLServe's boundary. If your backend deserialises a model artifact,
  > use `:erlang.binary_to_term(bin, [:safe])` — the unsafe form can exhaust the atom table and
  > construct arbitrary terms from a hostile file.

  ## What is checked

    * The path resolves inside the configured `:model_root` — `..` traversal and absolute paths
      pointing elsewhere are rejected, and symlinks are resolved before the check so a link cannot
      escape the root.
    * The file exists, is a regular file, and is readable.
    * The file is no larger than `:max_model_bytes`.
    * When a `:checksum` is configured, the file's digest matches.
  """

  alias MLServe.Error

  @typedoc "A digest algorithm and its expected lowercase hex value."
  @type checksum :: {:sha256 | :sha512, String.t()}

  @doc """
  Validates a configured model path and returns its absolute, symlink-resolved form.

  ## Parameters

    - `path`: the configured path, absolute or relative to `:model_root`
    - `opts`: `:root` (defaults to the configured model root), `:max_bytes`, `:checksum`

  ## Examples

      iex> MLServe.Security.validate_path("../../etc/passwd", root: "/srv/models")
      {:error, {:invalid_path, :outside_root}}
  """
  @spec validate_path(String.t(), keyword()) ::
          {:ok, String.t()} | {:error, {:invalid_path, atom()} | {:checksum_mismatch, String.t()}}
  def validate_path(path, opts \\ []) when is_binary(path) do
    root = opts |> Keyword.get(:root, MLServe.Config.model_root()) |> Path.expand()
    max_bytes = Keyword.get(opts, :max_bytes, MLServe.Config.max_model_bytes())

    with {:ok, resolved} <- resolve(path, root),
         :ok <- contained?(resolved, root),
         {:ok, size} <- regular_file(resolved),
         :ok <- within_size(size, max_bytes),
         :ok <- verify_checksum(resolved, Keyword.get(opts, :checksum)) do
      {:ok, resolved}
    end
  end

  @doc """
  Same as `validate_path/2` but raises `MLServe.Error` on failure.
  """
  @spec validate_path!(String.t(), keyword()) :: String.t()
  def validate_path!(path, opts \\ []) do
    case validate_path(path, opts) do
      {:ok, resolved} ->
        resolved

      {:error, reason} ->
        raise Error.new(:config, "invalid model path #{inspect(path)}: #{describe(reason)}",
                details: %{path: path}
              )
    end
  end

  @doc """
  Verifies a file against an expected digest.

  ## Parameters

    - `path`: absolute path to the file
    - `checksum`: `{:sha256 | :sha512, hex_string}`, or `nil` to skip

  ## Examples

      iex> MLServe.Security.verify_checksum("/nonexistent", nil)
      :ok
  """
  @spec verify_checksum(String.t(), checksum() | nil) ::
          :ok | {:error, {:checksum_mismatch, String.t()} | {:invalid_path, atom()}}
  def verify_checksum(_path, nil), do: :ok

  def verify_checksum(path, {algorithm, expected})
      when algorithm in [:sha256, :sha512] and is_binary(expected) do
    case digest(path, algorithm) do
      {:ok, actual} ->
        # Constant-time comparison: a digest check is an integrity check, but comparing with ==
        # on a hostile input is a needless timing side channel and the fix costs nothing.
        if secure_compare(actual, String.downcase(expected)) do
          :ok
        else
          {:error, {:checksum_mismatch, actual}}
        end

      error ->
        error
    end
  end

  @doc """
  Computes the lowercase hex digest of a file, streaming it rather than reading it into memory.

  Model artifacts are routinely gigabytes; `File.read!/1` followed by `:crypto.hash/2` would
  double peak memory at load time for no reason.
  """
  @spec digest(String.t(), :sha256 | :sha512) ::
          {:ok, String.t()} | {:error, {:invalid_path, atom()}}
  def digest(path, algorithm \\ :sha256) do
    path
    |> File.stream!(2 * 1024 * 1024)
    |> Enum.reduce(:crypto.hash_init(algorithm), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
    |> then(&{:ok, &1})
  rescue
    error in File.Error -> {:error, {:invalid_path, error.reason}}
  end

  @doc """
  Ensures a module is loaded and actually implements `MLServe.Model`.

  Checked at load time rather than assumed. A typo'd backend module would otherwise surface as an
  `UndefinedFunctionError` on the first prediction, long after the misconfiguration.
  """
  @spec validate_backend(module()) ::
          :ok | {:error, {:invalid_backend, :not_loaded | :not_a_model}}
  def validate_backend(backend) when is_atom(backend) do
    if Code.ensure_loaded?(backend) do
      if function_exported?(backend, :load, 1) and function_exported?(backend, :predict, 2) do
        :ok
      else
        {:error, {:invalid_backend, :not_a_model}}
      end
    else
      {:error, {:invalid_backend, :not_loaded}}
    end
  end

  def validate_backend(_), do: {:error, {:invalid_backend, :not_loaded}}

  # Private Functions

  defp resolve(path, root) do
    expanded =
      if Path.type(path) == :absolute, do: Path.expand(path), else: Path.expand(path, root)

    # Resolve symlinks before the containment check, otherwise a link inside the root pointing at
    # /etc would pass. `:file.read_link_all` fails on non-links, which is the common case.
    case File.stat(expanded) do
      {:ok, _} -> {:ok, real_path(expanded)}
      {:error, reason} -> {:error, {:invalid_path, reason}}
    end
  end

  defp real_path(path) do
    case :file.read_link_all(path) do
      {:ok, target} ->
        target = List.to_string(target)

        resolved =
          if Path.type(target) == :absolute,
            do: target,
            else: Path.expand(target, Path.dirname(path))

        real_path(resolved)

      {:error, _} ->
        path
    end
  end

  defp contained?(resolved, root) do
    if resolved == root or String.starts_with?(resolved, root <> "/") do
      :ok
    else
      {:error, {:invalid_path, :outside_root}}
    end
  end

  defp regular_file(path) do
    case File.stat(path) do
      {:ok, %File.Stat{type: :regular, size: size, access: access}}
      when access in [:read, :read_write] ->
        {:ok, size}

      {:ok, %File.Stat{type: :regular}} ->
        {:error, {:invalid_path, :not_readable}}

      {:ok, %File.Stat{}} ->
        {:error, {:invalid_path, :not_a_regular_file}}

      {:error, reason} ->
        {:error, {:invalid_path, reason}}
    end
  end

  defp within_size(_size, :infinity), do: :ok
  defp within_size(size, max) when size <= max, do: :ok
  defp within_size(_size, _max), do: {:error, {:invalid_path, :too_large}}

  defp secure_compare(a, b) when byte_size(a) == byte_size(b) do
    :crypto.hash_equals(a, b)
  end

  defp secure_compare(_a, _b), do: false

  defp describe({:invalid_path, :outside_root}), do: "resolves outside the configured :model_root"
  defp describe({:invalid_path, :too_large}), do: "exceeds the configured :max_model_bytes"
  defp describe({:invalid_path, reason}), do: "#{inspect(reason)}"
  defp describe({:checksum_mismatch, actual}), do: "checksum mismatch (got #{actual})"
end
