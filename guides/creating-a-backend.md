# Creating a Model Backend

A backend adapts an inference engine to MLServe. The contract is small — two required functions —
and this guide contains complete, runnable implementations for the runtimes people actually use.

MLServe ships no ML dependencies. These are guides rather than modules on purpose: the core stays
at one runtime dependency, and your backend tracks its engine's version rather than MLServe's.

## The contract

```elixir
defmodule MyApp.Backends.Minimal do
  @behaviour MLServe.Model

  @impl true
  def load(config), do: {:ok, build_state(config)}

  @impl true
  def predict(state, input), do: {:ok, run(state, input)}
end
```

`load/1` receives the model's `:config` keyword list with `:version` — and `:path` when one is
configured — already injected and validated. Whatever you return is the state passed to every
other callback. MLServe never inspects it.

### Optional callbacks

Detected with `function_exported?/3`, so implementing one is enough:

| Callback | Purpose |
| --- | --- |
| `batch_predict(state, inputs)` | Vectorised inference. Return results in input order, or `{:error, :not_supported}` to fall back to a mapped `predict/2`. |
| `unload(state)` | Release ports, files, NIF resources. Called on unload and reload. |
| `metadata(state)` | Map surfaced under `:metadata` in `MLServe.model_status/2`. Input shapes, label names, training run ids. |
| `capabilities()` | How MLServe may execute you. See below. |

### Declaring capabilities

This is the most important thing a backend declares, because it changes the execution strategy
entirely:

```elixir
@impl true
def capabilities, do: %{concurrency: :shared, load: :once}
```

| `:concurrency` | Effect |
| --- | --- |
| `:exclusive` (default) | State lives in pooled workers; one request at a time each. For anything not safe to call concurrently. |
| `:shared` | `predict/2` runs **in the calling process**, state read from `:persistent_term`. No worker processes, no message copies. |

| `:load` | Effect |
| --- | --- |
| `:once` (default) | `load/1` called once, state shared by all workers. Correct for NIF-resource handles. |
| `:per_worker` | `load/1` called per worker. Correct for ports and per-worker sessions. |

Choosing wrongly is the main way to leave performance on the floor. If `predict/2` is a pure
function or delegates to something already concurrency-safe, use `:shared`.

### Errors

Return `{:error, reason}` for expected failures. You do not need `try` to protect MLServe: every
callback is wrapped, and a raise, throw or exit becomes
`{:error, {:backend_error, %MLServe.BackendError{}}}` with the original exception and stacktrace
preserved.

> #### Model files are data {: .warning}
>
> MLServe never calls `:erlang.binary_to_term/1`, `Code.eval_*` or loads a NIF from a configured
> path. If your backend deserialises an artifact, use `:erlang.binary_to_term(bin, [:safe])` — the
> unsafe form can exhaust the atom table and construct arbitrary terms from a hostile file.

---

## Nx {: #nx}

`Nx.Serving` is already concurrency-safe and does its own batching, which makes `:shared` exactly
right — the alternative would copy tensors into a worker mailbox and back for no benefit.

```elixir
defmodule MyApp.Backends.NxServing do
  @moduledoc "Serves an Nx.Serving. Runs in the calling process."
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(config) do
    serving = Keyword.fetch!(config, :serving)

    case Keyword.get(config, :name) do
      # A named, supervised serving: batched_run/2 routes through Nx.Serving's own process,
      # which does cross-request batching for us.
      nil -> {:ok, {:inline, serving}}
      name -> {:ok, {:batched, name}}
    end
  end

  @impl true
  def predict({:inline, serving}, input), do: {:ok, Nx.Serving.run(serving, input)}
  def predict({:batched, name}, input), do: {:ok, Nx.Serving.batched_run(name, input)}

  @impl true
  def batch_predict(state, inputs) do
    # Nx.Serving batches internally; a stacked tensor is the efficient shape.
    {:ok, Enum.map(inputs, fn input -> elem(predict(state, input), 1) end)}
  end

  @impl true
  def metadata({kind, _}), do: %{serving: kind}
end
```

```elixir
config :ml_serve,
  models: [
    classifier: [
      backend: MyApp.Backends.NxServing,
      config: [name: MyApp.Serving]      # started in your own supervision tree
    ]
  ]
```

For a raw `Nx.Defn` function with no serving:

```elixir
defmodule MyApp.Backends.Nx do
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(config) do
    params = config |> Keyword.fetch!(:path) |> File.read!() |> Nx.deserialize()
    {:ok, params}
  end

  @impl true
  def predict(params, input) do
    {:ok, MyApp.Model.predict(params, Nx.tensor(input))}
  end

  @impl true
  def batch_predict(params, inputs) do
    batched = inputs |> Enum.map(&Nx.tensor/1) |> Nx.stack()
    results = MyApp.Model.predict(params, batched)
    {:ok, Enum.map(0..(length(inputs) - 1), &results[&1])}
  end
end
```

---

## Bumblebee {: #bumblebee}

Bumblebee produces an `Nx.Serving`, so the shape is the same. Load it in `load/1` and let MLServe
handle lifecycle, telemetry and versioning around it.

```elixir
defmodule MyApp.Backends.Bumblebee do
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(config) do
    repo = Keyword.fetch!(config, :repository)

    with {:ok, model} <- Bumblebee.load_model({:hf, repo}),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer({:hf, repo}) do
      serving =
        Bumblebee.Text.text_classification(model, tokenizer,
          compile: [batch_size: Keyword.get(config, :batch_size, 8), sequence_length: 128],
          defn_options: [compiler: EXLA]
        )

      {:ok, %{serving: serving, repository: repo}}
    end
  end

  @impl true
  def predict(%{serving: serving}, text) when is_binary(text) do
    {:ok, Nx.Serving.run(serving, text)}
  end

  @impl true
  def batch_predict(%{serving: serving}, texts) do
    {:ok, Nx.Serving.run(serving, texts)}
  end

  @impl true
  def metadata(%{repository: repo}), do: %{repository: repo}
end
```

Loading a Bumblebee model takes tens of seconds. MLServe's asynchronous loading is what keeps that
off your application's boot path — the model reports `:loading` and `MLServe.ready?/1` returns
`false` until it is genuinely servable, which is precisely what a readiness probe should see.

---

## ONNX via Ortex {: #onnx}

An ONNX Runtime session is a NIF resource. The handle is cheap to share, so `load: :once` is
right, but sessions are not always safe to call concurrently — so `:exclusive` with a pool is the
conservative and correct default.

```elixir
defmodule MyApp.Backends.ONNX do
  @behaviour MLServe.Model

  @impl true
  def capabilities do
    # :once — the session term is a resource handle. All workers share the underlying memory;
    # loading per worker would multiply GPU memory by the pool size.
    %{concurrency: :exclusive, load: :once}
  end

  @impl true
  def load(config) do
    path = Keyword.fetch!(config, :path)
    providers = Keyword.get(config, :providers, [:cpu])

    {:ok,
     %{
       session: Ortex.load(path, providers),
       inputs: Keyword.get(config, :inputs, ["input"]),
       version: Keyword.fetch!(config, :version)
     }}
  rescue
    error -> {:error, {:ortex_load_failed, Exception.message(error)}}
  end

  @impl true
  def predict(state, features) when is_map(features) do
    tensor = features |> encode() |> Nx.tensor(type: :f32) |> Nx.new_axis(0)
    {probabilities} = Ortex.run(state.session, {tensor})

    probability = probabilities |> Nx.squeeze() |> Nx.to_number()

    {:ok,
     %{
       prediction: if(probability > 0.5, do: :fraud, else: :legitimate),
       probability: Float.round(probability, 4)
     }}
  end

  @impl true
  def batch_predict(state, batch) do
    tensor = batch |> Enum.map(&encode/1) |> Nx.tensor(type: :f32)
    {probabilities} = Ortex.run(state.session, {tensor})

    results =
      probabilities
      |> Nx.to_flat_list()
      |> Enum.map(fn probability ->
        %{
          prediction: if(probability > 0.5, do: :fraud, else: :legitimate),
          probability: Float.round(probability, 4)
        }
      end)

    {:ok, results}
  end

  @impl true
  def metadata(state), do: %{inputs: state.inputs, model_version: state.version}

  defp encode(%{amount: amount, transaction_count_24h: count, failed_transactions_24h: failed}) do
    [amount / 10_000, count / 100, failed / 10]
  end
end
```

```elixir
config :ml_serve,
  model_root: "priv/models",
  models: [
    fraud_detection: [
      backend: MyApp.Backends.ONNX,
      path: "fraud.onnx",
      checksum: {:sha256, "ab12…"},
      workers: 4,
      batching: [max_size: 32, timeout: 10]
    ]
  ]
```

`:path` is validated against `:model_root` before your `load/1` sees it, and the checksum is
verified. See [Production Deployment](production-deployment.md).

> #### Runnable version {: .tip}
>
> [`examples/onnx`](https://github.com/jamesnjovu/ml_serve/tree/main/examples/onnx) is this
> backend as a script you can actually execute, against a committed 323-byte `.onnx` file:
> `elixir examples/onnx/fraud_detection.exs`. It also shows the checksum being enforced and a
> traversal path rejected, rather than only asserting that they are. Alongside it,
> `transformer.exs` runs a PyTorch-exported GPT-NeoX whose batch dimension is pinned to 1 —
> the case where a backend must *not* implement `batch_predict/2`.

---

## A Python model server over a port {: #python}

A port is stateful, single-conversation and not safe to share, so this is the case for
`load: :per_worker`: each worker owns its own OS process.

```elixir
defmodule MyApp.Backends.PythonPort do
  @behaviour MLServe.Model

  @impl true
  def capabilities do
    # Each worker owns an OS process. Sharing one port across a pool would interleave requests
    # on a single stdin/stdout conversation and return the wrong answers to the wrong callers.
    %{concurrency: :exclusive, load: :per_worker}
  end

  @impl true
  def load(config) do
    script = Keyword.fetch!(config, :script)
    python = Keyword.get(config, :python, System.find_executable("python3"))

    port =
      Port.open({:spawn_executable, python}, [
        :binary,
        :exit_status,
        {:packet, 4},
        {:args, [script, "--model", Keyword.fetch!(config, :path)]}
      ])

    receive do
      {^port, {:data, "ready"}} -> {:ok, port}
      {^port, {:exit_status, status}} -> {:error, {:python_exited, status}}
    after
      30_000 -> {:error, :python_startup_timeout}
    end
  end

  @impl true
  def predict(port, input) do
    Port.command(port, JSON.encode!(input))

    receive do
      {^port, {:data, response}} -> {:ok, JSON.decode!(response)}
      # An exit here becomes a BackendError with the status preserved. Pair with
      # restart_on_error: true so the worker is replaced with a fresh process.
      {^port, {:exit_status, status}} -> exit({:python_exited, status})
    after
      10_000 -> {:error, :timeout}
    end
  end

  @impl true
  def unload(port) do
    Port.close(port)
    :ok
  rescue
    ArgumentError -> :ok
  end
end
```

```elixir
models: [
  recommender: [
    backend: MyApp.Backends.PythonPort,
    path: "recommender.pkl",
    workers: 4,                 # 4 Python processes
    restart_on_error: true,     # a dead port must not be reused
    config: [script: "priv/python/serve.py"]
  ]
]
```

---

## A remote inference service

An HTTP call is safe to make concurrently, so `:shared` avoids a pointless pool. Bound the
concurrency with `:max_concurrency` instead, which needs no processes at all.

```elixir
defmodule MyApp.Backends.Remote do
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :shared, load: :once}

  @impl true
  def load(config), do: {:ok, %{url: Keyword.fetch!(config, :url)}}

  @impl true
  def predict(%{url: url}, input) do
    case Req.post(url, json: input, receive_timeout: 5_000) do
      {:ok, %{status: 200, body: body}} -> {:ok, body}
      {:ok, %{status: status}} -> {:error, {:http_error, status}}
      {:error, reason} -> {:error, reason}
    end
  end
end
```

```elixir
models: [remote_scorer: [backend: MyApp.Backends.Remote, max_concurrency: 64, config: [url: "..."]]]
```

---

## Testing your backend

Backends are plain modules — test them directly, then test the integration through MLServe:

```elixir
defmodule MyApp.Backends.ONNXTest do
  use ExUnit.Case, async: true

  test "predicts from a real model file" do
    {:ok, state} = MyApp.Backends.ONNX.load(path: "test/fixtures/fraud.onnx", version: "1.0.0")

    assert {:ok, %{prediction: prediction}} =
             MyApp.Backends.ONNX.predict(state, %{
               amount: 15_000,
               transaction_count_24h: 40,
               failed_transactions_24h: 9
             })

    assert prediction in [:fraud, :legitimate]
  end

  test "serves through MLServe" do
    name = :"test_#{System.unique_integer([:positive])}"

    {:ok, _} =
      MLServe.load_model(name,
        backend: MyApp.Backends.ONNX,
        path: "test/fixtures/fraud.onnx",
        workers: 2
      )

    on_exit(fn -> MLServe.unload_model(name) end)
    :ok = MLServe.await_ready(name)

    assert {:ok, _result} = MLServe.predict(name, %{amount: 100, transaction_count_24h: 1, failed_transactions_24h: 0})
  end
end
```

## Checklist

- [ ] `load/1` returns `{:ok, state}` or `{:error, reason}` — never raises for expected failures
- [ ] `capabilities/0` declares `:concurrency` and `:load` honestly
- [ ] `batch_predict/2` returns results **in input order and of equal length**, if implemented
- [ ] `unload/1` releases ports, files and resources, if you hold any
- [ ] `metadata/1` surfaces the model version or training run id, for `model_status/2`
- [ ] Any deserialisation of an artifact uses `[:safe]`
