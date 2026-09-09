# Getting Started

This guide takes you from an empty project to a supervised, instrumented model serving traffic.

## Install

```elixir
# mix.exs
def deps do
  [{:ml_serve, "~> 0.1.0"}]
end
```

MLServe starts its own supervision tree as an OTP application. There is nothing to add to your
own `application.ex`.

## 1. Write a backend

A backend is the adapter between MLServe and an inference engine. The contract is two functions:

```elixir
defmodule MyApp.FraudModel do
  @behaviour MLServe.Model

  @impl true
  def load(config) do
    {:ok, %{threshold: Keyword.fetch!(config, :threshold)}}
  end

  @impl true
  def predict(%{threshold: threshold}, %{amount: amount}) do
    probability = min(amount / 2000, 1.0)

    {:ok, %{
      prediction: if(probability > threshold, do: :fraud, else: :legitimate),
      probability: Float.round(probability, 2)
    }}
  end
end
```

`load/1` runs once when the model is registered and returns whatever state you want; `predict/2`
receives that state and one input. MLServe never looks inside either.

Real backends wrap `Nx.Serving`, an ONNX session, or a Python port — see
[Creating a Model Backend](creating-a-backend.md). The shape is the same.

## 2. Register the model

In configuration:

```elixir
# config/config.exs
config :ml_serve,
  models: [
    fraud_detection: [
      backend: MyApp.FraudModel,
      version: "1.0.0",
      workers: 4,
      config: [threshold: 0.7]
    ]
  ]
```

Or at runtime — the same validation and the same code path:

```elixir
MLServe.load_model(:fraud_detection,
  backend: MyApp.FraudModel,
  version: "1.0.0",
  workers: 4,
  config: [threshold: 0.7]
)
```

### Loading is asynchronous

`load_model/2` returns as soon as the model is *registered*. Loading happens in the background,
because a model that takes thirty seconds to memory-map should not hold up your application's
boot — and with several such models you could exceed the start timeout and fail to boot at all.

Until it is ready, predictions return `{:error, :model_not_ready}`:

```elixir
MLServe.ready?(:fraud_detection)          #=> false, briefly
MLServe.await_ready(:fraud_detection)     #=> :ok
```

A load that fails is retried with exponential backoff before the model is marked `:failed` —
model artifacts live on network mounts and volumes that attach late, and crashing on the first
failure would take the application down permanently for a problem that resolves in two seconds.

## 3. Predict

```elixir
iex> MLServe.predict(:fraud_detection, %{amount: 1500.50})
{:ok, %{prediction: :fraud, probability: 0.75}}

iex> MLServe.predict(:fraud_detection, %{amount: 10})
{:ok, %{prediction: :legitimate, probability: 0.01}}
```

Or a batch:

```elixir
iex> MLServe.batch_predict(:fraud_detection, [%{amount: 10}, %{amount: 20_000}])
{:ok, [%{prediction: :legitimate, ...}, %{prediction: :fraud, ...}]}
```

## 4. Look at it

```elixir
iex> MLServe.models()
[:fraud_detection]

iex> MLServe.model_status(:fraud_detection)
{:ok, %{
  name: :fraud_detection,
  version: "1.0.0",
  status: :ready,
  backend: MyApp.FraudModel,
  concurrency: :exclusive,
  workers: 4,
  in_flight: 0,
  requests: 2,
  errors: 0,
  default?: true,
  canary: nil,
  load_duration_ms: 1,
  ...
}}
```

`requests`, `errors` and `in_flight` come from atomic counters updated on the hot path. Reading
them asks no process and costs nothing.

## 5. Turn on visibility

The zero-dependency option, good for development:

```elixir
MLServe.Telemetry.Logger.attach(level: :info)
```

```text
[ml_serve] loaded :fraud_detection v1.0.0 (ok) in 1.2ms with 4 worker(s)
[ml_serve] :fraud_detection v1.0.0 ok in 0.42ms (inference 0.31ms, queue 0.02ms, batch 1)
```

For production, feed the events to LiveDashboard or a metrics reporter — see
[Telemetry](telemetry.md).

## 6. Wire it into Phoenix

```elixir
defmodule MyAppWeb.MLController do
  use MyAppWeb, :controller

  def predict(conn, params) do
    case MLServe.predict(:fraud_detection, params) do
      {:ok, result} -> json(conn, result)
      {:error, reason} -> conn |> put_status(:unprocessable_entity) |> json(%{error: inspect(reason)})
    end
  end
end
```

Nothing about MLServe is Phoenix-aware, and `predict/3` runs in the controller's own process, so
there is no extra hop. See [Phoenix Integration](phoenix-integration.md) for status-code mapping
and readiness probes.

## 7. Testing your application

Swap the backend for a static one in your test environment and your whole application exercises
its real code path against a known result:

```elixir
# config/test.exs
config :ml_serve,
  models: [
    fraud_detection: [
      backend: MLServe.Backend.Static,
      config: [result: %{prediction: :fraud, probability: 0.94}]
    ]
  ]
```

No model file, no ML runtime, no network.

## Where next

- [Architecture](architecture.md) — how the supervision tree and the hot path fit together
- [Concurrency](concurrency.md) — worker pools, shared backends, and picking `workers:`
- [Creating a Model Backend](creating-a-backend.md) — Nx, Bumblebee, ONNX and Python
- [Model Versioning](model-versioning.md) — canary rollout and zero-downtime upgrades
