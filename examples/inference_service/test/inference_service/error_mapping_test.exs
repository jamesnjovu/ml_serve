defmodule InferenceService.ErrorMappingTest do
  @moduledoc "The mapping is the part of this service most worth pinning down."

  use ExUnit.Case, async: true

  alias InferenceService.ErrorMapping

  test "maps every MLServe reason to the status a client can act on" do
    assert {404, %{error: "model_not_found"}} = ErrorMapping.to_http(:model_not_found)
    assert {503, %{error: "model_not_ready"}} = ErrorMapping.to_http(:model_not_ready)
    assert {504, %{error: "timeout"}} = ErrorMapping.to_http(:timeout)
    assert {429, %{error: "overloaded"}} = ErrorMapping.to_http(:overloaded)
    assert {422, %{error: "invalid_input"}} = ErrorMapping.to_http({:invalid_input, "bad"})
    assert {413, %{error: "batch_too_large"}} = ErrorMapping.to_http({:batch_too_large, 10})
    assert {503, %{error: "load_failed"}} = ErrorMapping.to_http({:load_failed, :corrupt})
  end

  test "marks transient failures retryable and permanent ones not" do
    for reason <- [:timeout, :overloaded, :model_not_ready] do
      assert {_status, %{retryable: true}} = ErrorMapping.to_http(reason)
    end

    for reason <- [:model_not_found, {:invalid_input, "bad"}, {:batch_too_large, 10}] do
      assert {_status, %{retryable: false}} = ErrorMapping.to_http(reason)
    end
  end

  test "a backend crash is a 500 that leaks nothing about the stacktrace" do
    error =
      MLServe.BackendError.new(
        backend: SomeModel,
        callback: {:predict, 2},
        kind: :error,
        reason: %RuntimeError{message: "tensor shape mismatch"},
        stacktrace: [],
        model: :fraud_detection,
        version: "1.0.0"
      )

    assert {500, body} = ErrorMapping.to_http({:backend_error, error})
    refute body.message =~ "tensor shape mismatch"
    refute body.retryable
  end
end
