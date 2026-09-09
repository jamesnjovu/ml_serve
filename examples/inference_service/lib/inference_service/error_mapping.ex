defmodule InferenceService.ErrorMapping do
  @moduledoc """
  Translates MLServe's error taxonomy into HTTP.

  This module is the reason this example exists. Every service wrapping an inference layer has
  to make these decisions, and getting them wrong is expensive in a way that is invisible until
  production: a 500 where a 429 belonged turns backpressure into an outage, because every client
  retry policy treats them differently.

  | MLServe reason | HTTP | Why |
  | -------------- | ---- | --- |
  | `:model_not_found` | 404 | The named model is not registered. A client bug. |
  | `:model_not_ready` | 503 | Registered but still loading, draining or failed. Try again. |
  | `:timeout` | 504 | The deadline passed. Upstream was too slow. |
  | `:overloaded` | 429 | At the admission limit. Backpressure, not failure — slow down. |
  | `{:invalid_input, _}` | 422 | Rejected by a `:preprocess` hook. The body was wrong. |
  | `{:batch_too_large, max}` | 413 | The batch exceeded `:max_batch_size`. |
  | `{:backend_error, _}` | 500 | The backend raised. Ours to fix, never the client's. |
  | `{:load_failed, _}` | 503 | The model could not be loaded at all. |

  `MLServe.Error.retryable?/1` already distinguishes transient failures from permanent ones, and
  it drives the `retryable` field below — so a client can decide whether to retry without
  parsing our error strings.
  """

  @spec to_http(term()) :: {100..599, map()}
  def to_http(reason) do
    {status, code, message} = classify(reason)

    {status, %{error: code, message: message, retryable: MLServe.Error.retryable?(reason)}}
  end

  defp classify(:model_not_found),
    do: {404, "model_not_found", "no model is registered under that name"}

  defp classify(:model_not_ready),
    do: {503, "model_not_ready", "the model is loading, draining or failed"}

  defp classify(:timeout),
    do: {504, "timeout", "inference did not complete before the deadline"}

  defp classify(:overloaded),
    do: {429, "overloaded", "the model is at its concurrency limit"}

  defp classify({:invalid_input, detail}),
    do: {422, "invalid_input", describe(detail)}

  defp classify({:batch_too_large, max}),
    do: {413, "batch_too_large", "the batch exceeded the limit of #{max} inputs"}

  defp classify({:load_failed, detail}),
    do: {503, "load_failed", "the model failed to load: #{describe(detail)}"}

  # Never leak a stacktrace to a client. It is in the logs and in telemetry, where it belongs.
  defp classify({:backend_error, error}) do
    require Logger
    Logger.error("backend error in #{inspect(error.backend)}: #{inspect(error.reason)}")
    {500, "inference_failed", "the model failed to produce a prediction"}
  end

  # A backend is free to return {:error, anything}. Treat an unrecognised reason as a rejection
  # by the model rather than a crash: it returned deliberately, it just did not use our vocabulary.
  defp classify(other), do: {422, "rejected", describe(other)}

  defp describe(detail) when is_binary(detail), do: detail
  defp describe(detail), do: inspect(detail)
end
