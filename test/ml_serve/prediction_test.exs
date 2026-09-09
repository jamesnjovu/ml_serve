defmodule MLServe.PredictionTest do
  use MLServe.Case, async: true

  alias MLServe.Test.Backends

  describe "predict/3" do
    test "returns the backend result" do
      name = load!(backend: Backends.Echo)

      assert MLServe.predict(name, %{amount: 1500.50}) == {:ok, %{amount: 1500.50}}
    end

    test "returns model_not_found for an unregistered model" do
      assert MLServe.predict(:definitely_not_loaded, 1) == {:error, :model_not_found}
    end

    test "routes to a pinned version" do
      name = load!(backend: Backends.Echo)
      {:ok, _} = MLServe.load_model(name, backend: Backends.Echo, version: "2.0.0")
      :ok = MLServe.await_ready({name, "2.0.0"})

      assert {:ok, :x} = MLServe.predict(name, :x, version: "2.0.0")
      assert MLServe.predict(name, :x, version: "9.9.9") == {:error, :model_not_found}
    end

    test "runs shared backends in the calling process" do
      name = load!(backend: Backends.Caller)

      assert {:ok, pid} = MLServe.predict(name, nil)
      assert pid == self()
    end

    test "runs exclusive backends in a worker, not the caller" do
      name = load!(backend: Backends.Pooled, workers: 1)

      assert {:ok, {_input, pid}} = MLServe.predict(name, :x)
      refute pid == self()
      assert pid == MLServe.Worker.whereis(name, "1.0.0", 0)
    end
  end

  describe "predict!/3" do
    test "returns the bare result" do
      name = load!(backend: Backends.Echo)

      assert MLServe.predict!(name, 42) == 42
    end

    test "raises MLServe.Error naming the model" do
      assert_raise MLServe.Error, ~r/:definitely_not_loaded is not loaded/, fn ->
        MLServe.predict!(:definitely_not_loaded, 1)
      end
    end

    test "raises MLServe.BackendError with the original exception preserved" do
      name = load!(backend: Backends.Crashing, workers: 1)

      error =
        assert_raise MLServe.BackendError, fn ->
          MLServe.predict!(name, :boom)
        end

      assert error.kind == :error
      assert %ArgumentError{message: "deliberate test failure"} = error.reason
      assert error.callback == {:predict, 2}
      assert error.backend == Backends.Crashing
      assert error.model == name
      assert error.stacktrace != []
    end
  end

  describe "preprocess and postprocess hooks" do
    test "preprocess transforms the input before dispatch" do
      name =
        load!(
          backend: Backends.Echo,
          preprocess: fn input -> {:ok, input * 10} end
        )

      assert MLServe.predict(name, 4) == {:ok, 40}
    end

    test "postprocess transforms the result" do
      name =
        load!(
          backend: Backends.Echo,
          postprocess: fn result -> {:ok, %{value: result}} end
        )

      assert MLServe.predict(name, 4) == {:ok, %{value: 4}}
    end

    test "hooks compose in order" do
      name =
        load!(
          backend: Backends.Echo,
          preprocess: fn n -> {:ok, n + 1} end,
          postprocess: fn n -> {:ok, n * 2} end
        )

      assert MLServe.predict(name, 1) == {:ok, 4}
    end

    test "preprocess rejects invalid input without reaching the backend" do
      tok = token()

      name =
        load!(
          backend: Backends.Counting,
          workers: 1,
          config: [token: tok],
          preprocess: fn
            n when is_integer(n) -> {:ok, n}
            other -> {:error, {:invalid_input, "expected an integer, got #{inspect(other)}"}}
          end
        )

      assert {:error, {:invalid_input, message}} = MLServe.predict(name, "nope")
      assert message =~ "expected an integer"
      assert Backends.count(tok, :predict) == 0
    end

    test "a bare error from a hook is tagged as invalid input" do
      name = load!(backend: Backends.Echo, preprocess: fn _ -> {:error, :nope} end)

      assert MLServe.predict(name, 1) == {:error, {:invalid_input, :nope}}
    end

    test "a hook returning a bare value is treated as the transformed value" do
      name = load!(backend: Backends.Echo, preprocess: fn n -> n * 3 end)

      assert MLServe.predict(name, 2) == {:ok, 6}
    end

    test "accepts an MFA tuple" do
      name = load!(backend: Backends.Echo, preprocess: {__MODULE__, :add, [5]})

      assert MLServe.predict(name, 1) == {:ok, 6}
    end

    test "hooks run in the calling process, not a worker" do
      parent = self()

      name =
        load!(
          backend: Backends.Pooled,
          workers: 1,
          preprocess: fn n ->
            send(parent, {:hook_ran_in, self()})
            {:ok, n}
          end
        )

      MLServe.predict(name, 1)

      assert_receive {:hook_ran_in, pid}
      assert pid == self()
    end
  end

  describe "backend error handling" do
    test "a raise becomes a structured backend error" do
      name = load!(backend: Backends.Crashing, workers: 1)

      assert {:error, {:backend_error, %MLServe.BackendError{} = error}} =
               MLServe.predict(name, :boom)

      assert %ArgumentError{} = error.reason
      assert Exception.message(error) =~ "predict/2 failed"
    end

    test "a throw becomes a structured backend error" do
      name = load!(backend: Backends.Crashing, workers: 1)

      assert {:error, {:backend_error, %MLServe.BackendError{kind: :throw, reason: :deliberate}}} =
               MLServe.predict(name, :throw)
    end

    test "an exit becomes a structured backend error" do
      name = load!(backend: Backends.Crashing, workers: 1)

      assert {:error, {:backend_error, %MLServe.BackendError{kind: :exit, reason: :deliberate}}} =
               MLServe.predict(name, :exit)
    end

    test "a malformed return value is reported with the offending term" do
      name = load!(backend: Backends.BadReturn)

      assert {:error, {:backend_error, error}} = MLServe.predict(name, 1)
      assert Exception.message(error) =~ "must return {:ok, result} or {:error, reason}"
      assert Exception.message(error) =~ ":not_a_tuple"
    end

    test "the worker survives a backend error by default" do
      name = load!(backend: Backends.Crashing, workers: 1)
      pid = MLServe.Worker.whereis(name, "1.0.0", 0)

      assert {:error, {:backend_error, _}} = MLServe.predict(name, :boom)

      assert MLServe.Worker.whereis(name, "1.0.0", 0) == pid
      assert Process.alive?(pid)
    end

    test "restart_on_error: true replaces the worker after a backend error" do
      name = load!(backend: Backends.Crashing, workers: 1, restart_on_error: true)
      pid = MLServe.Worker.whereis(name, "1.0.0", 0)

      assert {:error, {:backend_error, _}} = MLServe.predict(name, :boom)

      # The caller still gets the error; the worker is replaced with fresh state behind it.
      replacement =
        eventually(fn ->
          case MLServe.Worker.whereis(name, "1.0.0", 0) do
            nil -> nil
            ^pid -> nil
            other -> other
          end
        end)

      assert replacement != pid
      assert Process.alive?(replacement)
    end

    test "an errored prediction increments the error counter" do
      name = load!(backend: Backends.Crashing, workers: 1)

      MLServe.predict(name, :boom)

      assert {:ok, %{errors: 1, requests: 1, in_flight: 0}} = MLServe.model_status(name)
    end
  end

  describe "timeouts" do
    test "returns :timeout when inference outlives the deadline" do
      name = load!(backend: Backends.Slow, workers: 1, config: [delay: 300])

      assert MLServe.predict(name, :x, timeout: 30) == {:error, :timeout}
    end

    test "the deadline travels with the request so queued work is shed, not run" do
      tok = token()

      # One worker, deliberately slow: request two onwards queue behind the first.
      name =
        load!(
          backend: Backends.Counting,
          workers: 1,
          config: [token: tok, delay: 200],
          timeout: 2_000
        )

      occupier = Task.async(fn -> MLServe.predict(name, :first, timeout: 2_000) end)
      eventually(fn -> match?({:ok, %{in_flight: 1}}, MLServe.model_status(name)) end)

      queued =
        1..3
        |> Task.async_stream(fn _ -> MLServe.predict(name, :queued, timeout: 20) end,
          max_concurrency: 3
        )
        |> Enum.map(fn {:ok, result} -> result end)

      assert Enum.all?(queued, &(&1 == {:error, :timeout}))
      assert {:ok, :first} = Task.await(occupier, 2_000)

      # The point: the worker dropped the expired work instead of running it. Without deadline
      # checking it would have executed all four and taken 800ms to serve nobody.
      assert eventually(fn -> Backends.count(tok, :predict) == 1 end)
    end
  end

  describe "admission control" do
    test "rejects requests beyond max_concurrency with :overloaded" do
      name =
        load!(
          backend: Backends.Slow,
          workers: 2,
          max_concurrency: 2,
          config: [delay: 150],
          timeout: 2_000
        )

      results =
        1..8
        |> Task.async_stream(fn i -> MLServe.predict(name, i) end, max_concurrency: 8)
        |> Enum.map(fn {:ok, result} -> result end)

      assert {:error, :overloaded} in results
      assert Enum.count(results, &match?({:ok, _}, &1)) <= 2
    end

    test "the in-flight counter returns to zero after rejections" do
      name = load!(backend: Backends.Echo, max_concurrency: 1)

      for _ <- 1..20, do: MLServe.predict(name, 1)

      assert {:ok, %{in_flight: 0}} = MLServe.model_status(name)
    end

    test "in_flight is decremented even when a hook raises" do
      name = load!(backend: Backends.Echo, preprocess: fn _ -> raise "hook exploded" end)

      assert_raise RuntimeError, "hook exploded", fn -> MLServe.predict(name, 1) end

      assert {:ok, %{in_flight: 0, errors: 1}} = MLServe.model_status(name)
    end
  end

  @doc false
  def add(n, amount), do: {:ok, n + amount}
end
