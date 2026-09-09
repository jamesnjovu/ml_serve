# Models declared in config/test.exs load asynchronously, exactly as they do in production.
# Waiting once here is cheaper and less flaky than waiting in every test.
:ok = MLServe.await_ready()

ExUnit.start()
