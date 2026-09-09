defmodule MLServe.ErrorsTest do
  use ExUnit.Case, async: true

  doctest MLServe.Error

  alias MLServe.BackendError
  alias MLServe.Error

  describe "wrap/2" do
    test "names the model in the message" do
      error = Error.wrap(:model_not_found, model: :fraud)

      assert error.type == :model_not_found
      assert Exception.message(error) == "model :fraud is not loaded"
      assert error.details == %{model: :fraud}
    end

    test "handles a missing model name" do
      assert Exception.message(Error.wrap(:model_not_found)) =~ "(unknown)"
    end

    test "maps every documented reason to a type" do
      for {reason, type} <- [
            {:model_not_found, :model_not_found},
            {:model_not_ready, :model_not_ready},
            {:timeout, :timeout},
            {:overloaded, :overloaded},
            {{:invalid_input, "bad"}, :invalid_input},
            {{:batch_too_large, 10}, :batch_too_large},
            {{:load_failed, :enoent}, :load_failed},
            {{:backend_error, :boom}, :backend_error}
          ] do
        assert Error.wrap(reason).type == type, "#{inspect(reason)} did not map to #{type}"
      end
    end

    test "records the limit for a batch that is too large" do
      error = Error.wrap({:batch_too_large, 64})

      assert error.details.max_batch_size == 64
      assert Exception.message(error) =~ "maximum size of 64"
    end

    test "passes a BackendError through unchanged" do
      backend_error =
        BackendError.new(
          backend: Foo,
          callback: {:predict, 2},
          kind: :error,
          reason: %RuntimeError{message: "x"}
        )

      assert Error.wrap({:backend_error, backend_error}) == backend_error
      assert Error.wrap(backend_error) == backend_error
    end

    test "an already-wrapped error gains details but keeps its own" do
      original = Error.new(:timeout, "timed out", details: %{model: :a})
      merged = Error.wrap(original, model: :b, version: "1.0.0")

      assert merged.details.model == :a
      assert merged.details.version == "1.0.0"
    end

    test "falls back to :unknown for an unrecognised reason" do
      assert Error.wrap(:something_else).type == :unknown
    end
  end

  describe "retryable?/1" do
    test "transient conditions are retryable" do
      for reason <- [:timeout, :overloaded, :model_not_ready] do
        assert Error.retryable?(Error.wrap(reason)), "#{reason} should be retryable"
        assert Error.retryable?(reason), "bare #{reason} should be retryable"
      end
    end

    test "permanent conditions are not" do
      for reason <- [
            :model_not_found,
            {:invalid_input, "bad"},
            {:batch_too_large, 1},
            {:load_failed, :enoent}
          ] do
        refute Error.retryable?(Error.wrap(reason)), "#{inspect(reason)} should not be retryable"
      end
    end

    test "a backend exception is never retryable" do
      error =
        BackendError.new(
          backend: Foo,
          callback: {:predict, 2},
          kind: :error,
          reason: %RuntimeError{message: "x"}
        )

      refute Error.retryable?(error)
    end
  end

  describe "MLServe.Error as an exception" do
    test "can be raised and rescued" do
      assert_raise MLServe.Error, "model :x is not loaded", fn ->
        raise Error.wrap(:model_not_found, model: :x)
      end
    end
  end

  describe "MLServe.BackendError" do
    test "builds a message naming the backend and callback" do
      error =
        BackendError.new(
          backend: MyApp.Model,
          callback: {:predict, 2},
          kind: :error,
          reason: %ArgumentError{message: "bad shape"},
          stacktrace: []
        )

      message = Exception.message(error)
      assert message =~ "MyApp.Model.predict/2 failed"
      assert message =~ "bad shape"
    end

    test "formats a throw" do
      error =
        BackendError.new(
          backend: MyApp.Model,
          callback: {:predict, 2},
          kind: :throw,
          reason: :nope
        )

      assert Exception.message(error) =~ "throw"
    end

    test "can be raised and rescued" do
      error =
        BackendError.new(
          backend: MyApp.Model,
          callback: {:load, 1},
          kind: :error,
          reason: %RuntimeError{message: "x"}
        )

      assert_raise MLServe.BackendError, fn -> raise error end
    end
  end
end
