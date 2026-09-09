# Phoenix Integration

Phoenix is **not** a dependency of MLServe, and does not need to be. `MLServe.predict/3` runs in
the calling process, so a controller action calls it the way it would call any function — no
adapter, no plug, no extra hop.

This guide covers the parts that are worth getting right: error mapping, readiness probes,
metrics, and LiveView.

## A controller

```elixir
defmodule MyAppWeb.MLController do
  use MyAppWeb, :controller

  def predict(conn, params) do
    case MLServe.predict(:fraud_detection, params) do
      {:ok, result} ->
        json(conn, result)

      {:error, reason} ->
        conn
        |> put_status(status_for(reason))
        |> json(%{error: message_for(reason)})
    end
  end

  # Map MLServe reasons onto the HTTP semantics a client can actually act on. Collapsing
  # everything to 500 tells the caller nothing and makes retries either useless or harmful.
  defp status_for({:invalid_input, _}), do: :unprocessable_entity   # 422 — fix the request
  defp status_for({:batch_too_large, _}), do: :payload_too_large    # 413
  defp status_for(:model_not_found), do: :not_found                 # 404
  defp status_for(:model_not_ready), do: :service_unavailable       # 503 — retry shortly
  defp status_for(:overloaded), do: :too_many_requests              # 429 — back off
  defp status_for(:timeout), do: :gateway_timeout                   # 504
  defp status_for(_), do: :internal_server_error                    # 500 — our bug

  defp message_for(reason), do: Exception.message(MLServe.Error.wrap(reason))
end
```

```elixir
# router.ex
scope "/api", MyAppWeb do
  pipe_through :api
  post "/predict/fraud", MLController, :predict
end
```

`MLServe.Error.wrap/2` turns any reason into a struct with a human-readable message, so you do not
have to write one per branch.

### Retry-After

For `:overloaded` and `:model_not_ready`, a `Retry-After` header turns a rejection into a usable
instruction:

```elixir
{:error, reason} when reason in [:overloaded, :model_not_ready] ->
  conn
  |> put_resp_header("retry-after", "1")
  |> put_status(status_for(reason))
  |> json(%{error: message_for(reason)})
```

`MLServe.Error.retryable?/1` tells you which reasons deserve this.

## Validating input

Put validation in a `preprocess` hook rather than the controller, and every caller gets it — the
HTTP endpoint, an Oban job, an IEx session:

```elixir
config :ml_serve,
  models: [
    fraud_detection: [
      backend: MyApp.Backends.ONNX,
      preprocess: {MyApp.Fraud.Input, :validate, []}
    ]
  ]
```

```elixir
defmodule MyApp.Fraud.Input do
  @required ~w(amount transaction_count_24h failed_transactions_24h)

  def validate(params) when is_map(params) do
    case Enum.reject(@required, &is_number(params[&1] || params[String.to_existing_atom(&1)])) do
      [] -> {:ok, normalise(params)}
      missing -> {:error, {:invalid_input, "missing or non-numeric: #{Enum.join(missing, ", ")}"}}
    end
  end

  def validate(other), do: {:error, {:invalid_input, "expected a map, got #{inspect(other)}"}}

  defp normalise(params), do: Map.new(params, fn {k, v} -> {to_atom(k), v} end)
  defp to_atom(k) when is_atom(k), do: k
  defp to_atom(k) when is_binary(k), do: String.to_existing_atom(k)
end
```

Hooks run in the caller, before dispatch, so an invalid request never occupies a worker.

> #### Do not call `String.to_atom/1` on request params {: .warning}
>
> Atoms are never garbage collected, and a hostile client can exhaust the atom table with unique
> keys. `String.to_existing_atom/1` inside a `rescue`, or keeping string keys throughout, are the
> safe options.

## Health and readiness probes

Models load asynchronously, so an instance can be *up* long before it can *serve*. That is what
readiness probes are for:

```elixir
defmodule MyAppWeb.HealthController do
  use MyAppWeb, :controller

  # Liveness: is the VM up? Never depends on models — a failing model should not cause the
  # orchestrator to kill and restart a healthy node in a loop.
  def alive(conn, _params), do: send_resp(conn, 200, "ok")

  # Readiness: should this instance receive traffic?
  def ready(conn, _params) do
    if MLServe.ready?() do
      json(conn, %{status: "ready", models: MLServe.models()})
    else
      conn |> put_status(:service_unavailable) |> json(%{status: "loading", models: statuses()})
    end
  end

  defp statuses do
    Map.new(MLServe.models(), fn name ->
      {:ok, status} = MLServe.model_status(name)
      {name, %{version: status.version, status: status.status}}
    end)
  end
end
```

```elixir
scope "/health", MyAppWeb do
  get "/live", HealthController, :alive
  get "/ready", HealthController, :ready
end
```

Keeping liveness independent of model state is the important part. Tying them together means a
model that cannot load causes an endless kill-restart cycle instead of one clearly unhealthy
instance you can inspect.

## Metrics and LiveDashboard

```elixir
defmodule MyAppWeb.Telemetry do
  use Supervisor
  import Telemetry.Metrics

  # ...

  def metrics do
    [
      summary("phoenix.endpoint.stop.duration", unit: {:native, :millisecond}),
      summary("my_app.repo.query.total_time", unit: {:native, :millisecond})
    ] ++ MLServe.Telemetry.Metrics.metrics()
  end
end
```

```elixir
# router.ex
import Phoenix.LiveDashboard.Router

scope "/" do
  pipe_through :browser
  live_dashboard "/dashboard", metrics: MyAppWeb.Telemetry
end
```

Inference latency, throughput, error rate, cache hit rate and batch sizes appear on the Metrics
tab, tagged by model and version. Requires the optional `:telemetry_metrics` dependency.

### Correlating with requests

To attach the model version to your request logs:

```elixir
:telemetry.attach("ml-serve-logger-metadata", [:ml_serve, :prediction, :stop], fn _e, m, meta, _c ->
  Logger.metadata(
    ml_model: meta.model,
    ml_version: meta.version,
    ml_duration_ms: System.convert_time_unit(m.duration, :native, :millisecond)
  )
end, nil)
```

Because the handler runs inline in the request process, `Logger.metadata/1` lands on the right
process — which would not work from a separate reporter.

## LiveView

`predict/3` is synchronous. In a LiveView, run it under `assign_async/3` or `start_async/3` so the
socket process stays responsive:

```elixir
defmodule MyAppWeb.FraudLive do
  use MyAppWeb, :live_view

  def mount(_params, _session, socket) do
    {:ok, assign(socket, form: to_form(%{}), result: nil)}
  end

  def handle_event("score", %{"transaction" => params}, socket) do
    {:noreply,
     socket
     |> assign(result: nil)
     |> start_async(:score, fn -> MLServe.predict(:fraud_detection, params) end)}
  end

  def handle_async(:score, {:ok, {:ok, result}}, socket) do
    {:noreply, assign(socket, result: result)}
  end

  def handle_async(:score, {:ok, {:error, reason}}, socket) do
    {:noreply, put_flash(socket, :error, Exception.message(MLServe.Error.wrap(reason)))}
  end

  def handle_async(:score, {:exit, reason}, socket) do
    {:noreply, put_flash(socket, :error, "inference crashed: #{inspect(reason)}")}
  end
end
```

A model status page is a natural LiveView too:

```elixir
def mount(_params, _session, socket) do
  if connected?(socket), do: :timer.send_interval(1_000, :refresh)
  {:ok, assign_statuses(socket)}
end

def handle_info(:refresh, socket), do: {:noreply, assign_statuses(socket)}

defp assign_statuses(socket) do
  assign(socket, models: Enum.map(MLServe.models(), fn name ->
    {:ok, status} = MLServe.model_status(name)
    status
  end))
end
```

`model_status/2` reads ETS and atomic counters, so polling it every second across many connected
clients is cheap.

## Channels

```elixir
def handle_in("predict", params, socket) do
  case MLServe.predict(:fraud_detection, params, timeout: 2_000) do
    {:ok, result} -> {:reply, {:ok, result}, socket}
    {:error, reason} -> {:reply, {:error, %{reason: inspect(reason)}}, socket}
  end
end
```

A channel process is per-connection, so blocking it blocks only that client. Still pass an
explicit `:timeout` — a channel that hangs for the default is a connection that appears frozen.

## Testing

Swap in a static backend and your controller tests exercise the real path with a known result:

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

```elixir
test "POST /api/predict/fraud", %{conn: conn} do
  conn = post(conn, ~p"/api/predict/fraud", %{amount: 1500.50})

  assert %{"prediction" => "fraud", "probability" => 0.94} = json_response(conn, 200)
end
```

For the error path, load a failing model under a unique name in the test itself:

```elixir
test "returns 503 while a model is loading", %{conn: conn} do
  name = :"loading_#{System.unique_integer([:positive])}"
  MLServe.load_model(name, backend: MyApp.SlowLoadingBackend)
  on_exit(fn -> MLServe.unload_model(name) end)

  assert MLServe.predict(name, %{}) == {:error, :model_not_ready}
end
```
