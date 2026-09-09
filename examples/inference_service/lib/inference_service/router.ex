defmodule InferenceService.Router do
  @moduledoc """
  A JSON inference API over MLServe.

  The interesting part is not the routing, it is `InferenceService.ErrorMapping`: turning
  MLServe's error taxonomy into the right HTTP status codes. Getting that mapping right is what
  lets a client — or a load balancer, or a retry policy — behave sensibly without knowing
  anything about MLServe.
  """

  use Plug.Router

  alias InferenceService.ErrorMapping

  plug(Plug.Logger)
  plug(:match)

  plug(Plug.Parsers,
    parsers: [:json],
    pass: ["application/json"],
    json_decoder: Jason
  )

  plug(:dispatch)

  # ── Readiness ────────────────────────────────────────────────────────────────────────────────

  # The probe a load balancer should poll. MLServe.ready?/1 with no argument reports whether
  # *every* declared model has finished loading, which is precisely the condition under which
  # this node should receive traffic. Models load asynchronously, so this flips to 200 on its
  # own once they are up — no restart, no readiness gate in the deployment config.
  get "/health" do
    if MLServe.ready?() do
      send_json(conn, 200, %{status: "ready", models: MLServe.models()})
    else
      send_json(conn, 503, %{status: "loading", models: model_statuses()})
    end
  end

  # ── Introspection ────────────────────────────────────────────────────────────────────────────

  get "/models" do
    send_json(conn, 200, %{models: model_statuses()})
  end

  get "/models/:name" do
    with {:ok, name} <- known_model(name),
         {:ok, status} <- MLServe.model_status(name) do
      send_json(conn, 200, describe(status))
    else
      {:error, reason} -> send_error(conn, reason)
    end
  end

  # ── Inference ────────────────────────────────────────────────────────────────────────────────

  # POST /predict/fraud_detection
  # POST /predict/fraud_detection?version=2.0.0   — pin a version, bypassing canary routing
  # POST /predict/fraud_detection?cache=false     — override the model's cache setting
  post "/predict/:name" do
    with {:ok, name} <- known_model(name),
         {:ok, opts} <- predict_opts(conn) do
      case MLServe.predict(name, conn.body_params, opts) do
        {:ok, result} -> send_json(conn, 200, %{result: result})
        {:error, reason} -> send_error(conn, reason)
      end
    else
      {:error, reason} -> send_error(conn, reason)
    end
  end

  # POST /predict/sentiment/batch  with  {"inputs": [...]}
  post "/predict/:name/batch" do
    with {:ok, name} <- known_model(name),
         {:ok, inputs} <- fetch_inputs(conn.body_params),
         {:ok, opts} <- predict_opts(conn) do
      case MLServe.batch_predict(name, inputs, opts) do
        {:ok, results} -> send_json(conn, 200, %{results: results, count: length(results)})
        {:error, reason} -> send_error(conn, reason)
      end
    else
      {:error, reason} -> send_error(conn, reason)
    end
  end

  match _ do
    send_json(conn, 404, %{error: "not_found", message: "no such route"})
  end

  # ── Helpers ──────────────────────────────────────────────────────────────────────────────────

  # String.to_existing_atom/1 rather than to_atom/1: a value from an HTTP path must never be able
  # to grow the atom table. A name no module has ever mentioned simply is not a model.
  defp known_model(name) do
    {:ok, String.to_existing_atom(name)}
  rescue
    ArgumentError -> {:error, :model_not_found}
  end

  defp fetch_inputs(%{"inputs" => inputs}) when is_list(inputs), do: {:ok, inputs}

  defp fetch_inputs(_params),
    do: {:error, {:invalid_input, "expected a JSON body with an \"inputs\" array"}}

  defp predict_opts(conn) do
    conn = Plug.Conn.fetch_query_params(conn)

    opts =
      []
      |> maybe_put(:version, conn.query_params["version"])
      |> maybe_put(:cache, parse_cache(conn.query_params["cache"]))
      |> maybe_put(:timeout, parse_int(conn.query_params["timeout"]))

    {:ok, opts}
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)

  defp parse_cache("true"), do: true
  defp parse_cache("false"), do: false
  defp parse_cache(_), do: nil

  defp parse_int(nil), do: nil

  defp parse_int(value) do
    case Integer.parse(value) do
      {int, ""} -> int
      _ -> nil
    end
  end

  defp model_statuses do
    for name <- MLServe.models(),
        {:ok, status} <- [MLServe.model_status(name)],
        do: describe(status)
  end

  defp describe(status) do
    %{
      name: status.name,
      version: status.version,
      status: status.status,
      backend: inspect(status.backend),
      concurrency: status.concurrency,
      workers: status.workers,
      requests: status.requests,
      errors: status.errors,
      in_flight: status.in_flight,
      default: status.default?,
      canary: describe_canary(status.canary),
      metadata: status.metadata
    }
  end

  defp describe_canary(nil), do: nil
  defp describe_canary({version, percent}), do: %{version: version, percent: percent}
  defp describe_canary(other), do: inspect(other)

  defp send_error(conn, reason) do
    {status, body} = ErrorMapping.to_http(reason)

    conn
    |> maybe_retry_after(status)
    |> send_json(status, body)
  end

  # 429 and 503 without a Retry-After are a client's invitation to hammer you harder.
  defp maybe_retry_after(conn, status) when status in [429, 503],
    do: Plug.Conn.put_resp_header(conn, "retry-after", "1")

  defp maybe_retry_after(conn, _status), do: conn

  defp send_json(conn, status, body) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(status, Jason.encode!(body))
  end
end
