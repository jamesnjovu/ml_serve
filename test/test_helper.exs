# Call counters for the test backends. :public so backends can write from any process, named so
# they need no plumbing. Keys are per-test tokens, which is what keeps the suite async-safe.
:ets.new(:ml_serve_test_calls, [:set, :public, :named_table, write_concurrency: true])

ExUnit.start(capture_log: true)
