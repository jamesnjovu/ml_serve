# Error handling — every failure MLServe can hand you, and what to do about each.
#
#     elixir examples/scripts/07_error_handling.exs
#
# Covers: the full error taxonomy triggered for real rather than described, telling retryable
# failures from permanent ones, the bang variants, input validation hooks, and swapping in a
# stub backend so your own tests never need a model.

Mix.install([
  # Running from a clone of the repo. Against the published package this is:
  #   {:ml_serve, "~> 0.1.0"}
  {:ml_serve, path: Path.expand("../..", __DIR__)}
])

section = fn title ->
  IO.puts("\n── " <> title <> " " <> String.duplicate("─", max(0, 62 - String.length(title))))
end

# Note: this matches {:backend_error, error} and reads error.reason rather than pattern-matching
# %MLServe.BackendError{}. In a .exs script the whole file is expanded before Mix.install has run,
# so a struct pattern from an installed dependency cannot be resolved. Inside a normal mix project
# — including examples/inference_service — the struct pattern is fine.
show = fn label, result ->
  reason =
    case result do
      {:error, {:backend_error, error}} ->
        "{:backend_error, #{inspect(error.reason)}}"

      {:error, reason} ->
        inspect(reason)

      {:ok, value} ->
        "{:ok, #{inspect(value)}}"
    end

  retryable =
    case result do
      {:error, reason} -> if MLServe.Error.retryable?(reason), do: "retry", else: "do not retry"
      _ -> "—"
    end

  reason =
    if String.length(reason) > 46, do: String.slice(reason, 0, 43) <> "...", else: reason

  IO.puts("    #{String.pad_trailing(label, 22)} #{String.pad_trailing(reason, 46)} #{retryable}")
end

defmodule SlowModel do
  @moduledoc "Slow to load and slow to predict, so readiness and timeout errors are reachable."
  @behaviour MLServe.Model

  @impl true
  def capabilities, do: %{concurrency: :exclusive, load: :once}

  @impl true
  def load(config) do
    Process.sleep(Keyword.get(config, :load_delay, 0))
    {:ok, Keyword.get(config, :predict_delay, 0)}
  end

  @impl true
  def predict(_delay, :raise), do: raise(RuntimeError, "tensor shape mismatch")

  def predict(delay, input) do
    if delay > 0, do: Process.sleep(delay)
    {:ok, input}
  end
end

IO.puts("""
  Every MLServe function returns {:ok, result} or {:error, reason}. The reasons below are all
  triggered for real — nothing here is a description of what would happen.
""")

section.("The taxonomy")

IO.puts("    #{String.pad_trailing("call", 22)} #{String.pad_trailing("reason", 46)} verdict")
IO.puts("    #{String.duplicate("─", 74)}")

# :model_not_found — nothing registered under that name, or not at that version.
show.("unknown model", MLServe.predict(:no_such_model, :input))

# :model_not_ready — registered, but still loading, draining, or failed. load_model/2 returns as
# soon as the model is registered; loading continues in the background.
{:ok, _} =
  MLServe.load_model(:still_loading, backend: SlowModel, config: [load_delay: 400])

show.("still loading", MLServe.predict(:still_loading, :input))

# :timeout — the deadline passed. The deadline travels with the request, so a worker that picks
# up already-expired work drops it rather than running it.
{:ok, _} =
  MLServe.load_model(:slow, backend: SlowModel, workers: 1, config: [predict_delay: 200])

:ok = MLServe.await_ready(:slow)
show.("deadline exceeded", MLServe.predict(:slow, :input, timeout: 20))

# :overloaded — the model is at its :max_concurrency limit. Admission happens in the caller, so
# a rejection costs one ETS read rather than a queue slot.
{:ok, _} =
  MLServe.load_model(:limited,
    backend: SlowModel,
    workers: 1,
    max_concurrency: 1,
    config: [predict_delay: 200]
  )

:ok = MLServe.await_ready(:limited)
Task.async(fn -> MLServe.predict(:limited, :input) end)
Process.sleep(20)
show.("at capacity", MLServe.predict(:limited, :input))

# {:invalid_input, reason} — rejected by a :preprocess hook before any worker was involved.
{:ok, _} =
  MLServe.load_model(:validated,
    backend: SlowModel,
    preprocess: fn
      %{amount: amount} = input when is_number(amount) -> {:ok, input}
      _other -> {:error, :amount_must_be_a_number}
    end
  )

:ok = MLServe.await_ready(:validated)
show.("rejected by hook", MLServe.predict(:validated, %{amount: "not a number"}))

# {:batch_too_large, max} — the list exceeded :max_batch_size. A guard against one caller
# turning a single request into an unbounded amount of work.
{:ok, _} = MLServe.load_model(:small_batches, backend: SlowModel, max_batch_size: 10)
:ok = MLServe.await_ready(:small_batches)
show.("oversized batch", MLServe.batch_predict(:small_batches, Enum.to_list(1..50)))

# {:backend_error, %MLServe.BackendError{}} — the backend raised. MLServe catches it at the
# backend boundary and preserves the exception, the stacktrace and which callback it came from.
show.("backend raised", MLServe.predict(:slow, :raise))

section.("Retryable or not")

IO.puts("""
    :timeout, :overloaded and :model_not_ready are transient — the same request may well succeed
    a moment later, and MLServe.Error.retryable?/1 says so. Everything else is a bug in the
    request or in the model, and retrying it only burns capacity.

    Reach for retryable?/1 rather than matching the atoms yourself; new transient reasons get
    added to it, not to your case statement.\
""")

section.("Inspecting a backend failure")

{:error, {:backend_error, error}} = MLServe.predict(:slow, :raise)

IO.puts("""
    backend      #{inspect(error.backend)}
    callback     #{inspect(error.callback)}
    kind         #{inspect(error.kind)}
    reason       #{inspect(error.reason)}
    model        #{inspect(error.model)} v#{error.version}
    top frame    #{error.stacktrace |> hd() |> then(fn {m, f, a, _} -> "#{inspect(m)}.#{f}/#{if(is_list(a), do: length(a), else: a)}" end)}

    The worker was not restarted — one bad input should not evict a model that took thirty
    seconds to load. Pass restart_on_error: true if the backend keeps state that a raise may
    have corrupted.\
""")

section.("Bang variants")

# predict!/3 raises MLServe.Error instead of returning a tuple. Use it where a failure is
# genuinely exceptional and you would only re-raise anyway.
try do
  MLServe.predict!(:no_such_model, :input)
rescue
  error in MLServe.Error ->
    IO.puts("    rescued      #{inspect(error.__struct__)}")
    IO.puts("    type         #{inspect(error.type)}")
    IO.puts("    message      #{Exception.message(error)}")
end

section.("Testing your own code without a model")

# MLServe.Backend.Static answers every prediction with a fixed value. Point config/test.exs at
# it and the system under test exercises real routing, caching and telemetry with no model file,
# no ML runtime and no GPU.
{:ok, _} =
  MLServe.load_model(:fraud_detection,
    backend: MLServe.Backend.Static,
    config: [result: %{prediction: :fraud, probability: 0.94}]
  )

:ok = MLServe.await_ready(:fraud_detection)
show.("stubbed success", MLServe.predict(:fraud_detection, %{anything: true}))

# And the error path, which is the half usually left untested.
{:ok, _} =
  MLServe.load_model(:flaky,
    backend: MLServe.Backend.Static,
    config: [error: :upstream_unavailable]
  )

:ok = MLServe.await_ready(:flaky)
show.("stubbed failure", MLServe.predict(:flaky, %{anything: true}))

for model <- MLServe.models(), do: MLServe.unload_model(model)
