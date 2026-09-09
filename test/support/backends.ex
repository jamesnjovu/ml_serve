defmodule MLServe.Test.Backends do
  @moduledoc """
  Model backends used by MLServe's own test suite.

  MLServe's tests must never require a multi-gigabyte model file or a Rust toolchain, so every
  behaviour the runtime cares about — slowness, crashing, refusing to load, native batching,
  per-worker state — is represented by a tiny deterministic backend here.

  Backends that need to record what happened write to the `:ml_serve_test_calls` ETS table
  created in `test_helper.exs`, keyed by a per-test token, so tests stay `async: true`.
  """

  @table :ml_serve_test_calls

  @doc "Records `n` calls against `token` and returns the new count."
  @spec record(term(), atom(), pos_integer()) :: pos_integer()
  def record(token, kind \\ :predict, n \\ 1) do
    :ets.update_counter(@table, {token, kind}, {2, n}, {{token, kind}, 0})
  end

  @doc "Returns how many calls of `kind` were recorded against `token`."
  @spec count(term(), atom()) :: non_neg_integer()
  def count(token, kind \\ :predict) do
    case :ets.lookup(@table, {token, kind}) do
      [{_key, count}] -> count
      [] -> 0
    end
  end

  defmodule Echo do
    @moduledoc "Shared-concurrency backend returning its input unchanged."
    @behaviour MLServe.Model

    @impl true
    def capabilities, do: %{concurrency: :shared, load: :once}

    @impl true
    def load(config), do: {:ok, Keyword.get(config, :tag, :echo)}

    @impl true
    def predict(_state, input), do: {:ok, input}

    @impl true
    def metadata(state), do: %{tag: state}
  end

  defmodule Caller do
    @moduledoc "Shared-concurrency backend returning the pid that ran the prediction."
    @behaviour MLServe.Model

    @impl true
    def capabilities, do: %{concurrency: :shared, load: :once}

    @impl true
    def load(_config), do: {:ok, nil}

    @impl true
    def predict(_state, _input), do: {:ok, self()}
  end

  defmodule Pooled do
    @moduledoc "Exclusive-concurrency backend returning `{input, worker_pid}`."
    @behaviour MLServe.Model

    @impl true
    def capabilities, do: %{concurrency: :exclusive, load: :once}

    @impl true
    def load(config), do: {:ok, Keyword.get(config, :tag, :pooled)}

    @impl true
    def predict(_state, input), do: {:ok, {input, self()}}
  end

  defmodule Slow do
    @moduledoc "Exclusive backend that sleeps `:delay` milliseconds before answering."
    @behaviour MLServe.Model

    @impl true
    def capabilities, do: %{concurrency: :exclusive, load: :once}

    @impl true
    def load(config), do: {:ok, Keyword.get(config, :delay, 50)}

    @impl true
    def predict(delay, input) do
      Process.sleep(delay)
      {:ok, input}
    end

    @impl true
    def batch_predict(delay, inputs) do
      Process.sleep(delay)
      {:ok, inputs}
    end
  end

  defmodule Crashing do
    @moduledoc "Raises on every prediction, to exercise backend-exception handling."
    @behaviour MLServe.Model

    @impl true
    def capabilities, do: %{concurrency: :exclusive, load: :once}

    @impl true
    def load(_config), do: {:ok, nil}

    @impl true
    def predict(_state, :exit), do: exit(:deliberate)
    def predict(_state, :throw), do: throw(:deliberate)
    def predict(_state, _input), do: raise(ArgumentError, "deliberate test failure")
  end

  defmodule BadReturn do
    @moduledoc "Returns a bare value instead of an ok/error tuple."
    @behaviour MLServe.Model

    # Violating the `predict/2` contract is the entire point of this backend, so the mismatch
    # dialyzer reports here is the fixture working as intended, not a defect to fix.
    @dialyzer {:nowarn_function, predict: 2}

    @impl true
    def capabilities, do: %{concurrency: :shared, load: :once}

    @impl true
    def load(_config), do: {:ok, nil}

    @impl true
    def predict(_state, _input), do: :not_a_tuple
  end

  defmodule FailingLoad do
    @moduledoc "Fails to load, unless `:succeed_after` attempts have been recorded."
    @behaviour MLServe.Model

    alias MLServe.Test.Backends

    @impl true
    def capabilities, do: %{concurrency: :shared, load: :once}

    @impl true
    def load(config) do
      token = Keyword.fetch!(config, :token)
      attempt = Backends.record(token, :load)

      case Keyword.get(config, :succeed_after) do
        nil -> {:error, :deliberate_load_failure}
        n when attempt > n -> {:ok, :recovered}
        _ -> {:error, :deliberate_load_failure}
      end
    end

    @impl true
    def predict(state, _input), do: {:ok, state}
  end

  defmodule Counting do
    @moduledoc "Records every predict and batch_predict call against a per-test token."
    @behaviour MLServe.Model

    alias MLServe.Test.Backends

    @impl true
    def capabilities, do: %{concurrency: :exclusive, load: :once}

    @impl true
    def load(config) do
      {:ok, {Keyword.fetch!(config, :token), Keyword.get(config, :delay, 0)}}
    end

    @impl true
    def predict({token, delay}, input) do
      Backends.record(token, :predict)
      if delay > 0, do: Process.sleep(delay)
      {:ok, input}
    end

    @impl true
    def batch_predict({token, delay}, inputs) do
      Backends.record(token, :batch)
      Backends.record(token, :batch_rows, length(inputs))
      if delay > 0, do: Process.sleep(delay)
      {:ok, Enum.map(inputs, &{:batched, &1})}
    end
  end

  defmodule PerWorker do
    @moduledoc "Loads independent state per worker, recording each load."
    @behaviour MLServe.Model

    alias MLServe.Test.Backends

    @impl true
    def capabilities, do: %{concurrency: :exclusive, load: :per_worker}

    @impl true
    def load(config) do
      token = Keyword.fetch!(config, :token)
      {:ok, {token, Backends.record(token, :load)}}
    end

    @impl true
    def predict({_token, instance}, _input), do: {:ok, instance}

    @impl true
    def unload({token, _instance}) do
      Backends.record(token, :unload)
      :ok
    end
  end

  defmodule Unloadable do
    @moduledoc "Records unload calls so lifecycle cleanup can be asserted."
    @behaviour MLServe.Model

    alias MLServe.Test.Backends

    @impl true
    def capabilities, do: %{concurrency: :shared, load: :once}

    @impl true
    def load(config), do: {:ok, Keyword.fetch!(config, :token)}

    @impl true
    def predict(_state, input), do: {:ok, input}

    @impl true
    def unload(token) do
      Backends.record(token, :unload)
      :ok
    end
  end

  defmodule Minimal do
    @moduledoc """
    The smallest possible backend: no `capabilities/0`, no optional callbacks.

    Exists to prove the documented defaults — `concurrency: :exclusive`, `load: :once` — really
    are what an undeclared backend gets.
    """
    @behaviour MLServe.Model

    @impl true
    def load(_config), do: {:ok, :minimal}

    @impl true
    def predict(_state, input), do: {:ok, input}
  end

  defmodule Nondeterministic do
    @moduledoc "Returns a fresh value on every call, so cache hits are observable."
    @behaviour MLServe.Model

    @impl true
    def capabilities, do: %{concurrency: :shared, load: :once}

    @impl true
    def load(_config), do: {:ok, nil}

    @impl true
    def predict(_state, _input), do: {:ok, System.unique_integer([:positive])}
  end
end
