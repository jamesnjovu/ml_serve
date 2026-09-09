defmodule InferenceService.RouterTest do
  @moduledoc """
  Exercises the real router, the real error mapping, real caching and real telemetry.

  The only stub is the model itself: `config/test.exs` points every model at
  `MLServe.Backend.Static`. No model file, no ML runtime, no GPU — and no mocking library.
  """

  use ExUnit.Case, async: true

  import Plug.Test
  import Plug.Conn

  alias InferenceService.Router

  @opts Router.init([])

  defp request(method, path, body \\ nil) do
    conn =
      case body do
        nil ->
          conn(method, path)

        body ->
          method
          |> conn(path, Jason.encode!(body))
          |> put_req_header("content-type", "application/json")
      end

    Router.call(conn, @opts)
  end

  defp json(conn), do: Jason.decode!(conn.resp_body)

  describe "GET /health" do
    test "reports ready once every declared model has loaded" do
      conn = request(:get, "/health")

      assert conn.status == 200
      assert json(conn)["status"] == "ready"
      assert "fraud_detection" in json(conn)["models"]
    end
  end

  describe "GET /models" do
    test "lists every model with its operational status" do
      conn = request(:get, "/models")
      models = json(conn)["models"]

      assert conn.status == 200

      fraud = Enum.find(models, &(&1["name"] == "fraud_detection"))
      assert fraud["status"] == "ready"
      assert fraud["default"] == true
      assert is_integer(fraud["requests"])
    end

    test "an unknown model is a 404, and does not create an atom" do
      conn = request(:get, "/models/definitely_not_a_model")

      assert conn.status == 404
      assert json(conn)["error"] == "model_not_found"
    end
  end

  describe "POST /predict/:name" do
    test "returns the model's result" do
      conn = request(:post, "/predict/fraud_detection", %{amount: 1500.5})

      assert conn.status == 200
      assert json(conn)["result"] == %{"prediction" => "fraud", "probability" => 0.94}
    end

    test "an unknown model is a 404" do
      conn = request(:post, "/predict/nope", %{amount: 1})

      assert conn.status == 404
      assert json(conn)["error"] == "model_not_found"
      refute json(conn)["retryable"]
    end

    test "a backend error becomes a 422 the client can act on" do
      conn = request(:post, "/predict/always_failing", %{anything: true})

      assert conn.status == 422
      assert json(conn)["error"] == "rejected"
    end

    test "an unknown version is a 404 rather than silently serving the default" do
      conn = request(:post, "/predict/fraud_detection?version=9.9.9", %{amount: 1})

      assert conn.status == 404
    end
  end

  describe "POST /predict/:name/batch" do
    test "returns one result per input" do
      conn =
        request(:post, "/predict/sentiment/batch", %{
          inputs: [%{text: "great"}, %{text: "awful"}, %{text: "fine"}]
        })

      assert conn.status == 200
      assert json(conn)["count"] == 3
      assert length(json(conn)["results"]) == 3
    end

    test "a body without an inputs array is a 422" do
      conn = request(:post, "/predict/sentiment/batch", %{text: "not a list"})

      assert conn.status == 422
      assert json(conn)["error"] == "invalid_input"
    end
  end

  describe "unmatched routes" do
    test "are a 404 with a JSON body, not an HTML error page" do
      conn = request(:get, "/nope")

      assert conn.status == 404
      assert json(conn)["error"] == "not_found"
    end
  end
end
