# Contributing to MLServe

Notes for humans and coding agents working **on** MLServe. If you are using MLServe in an
application, read [`usage-rules.md`](usage-rules.md) instead.

## Before you push

```bash
mix lint      # format --check-formatted + compile --warnings-as-errors + credo --strict
mix test
mix dialyzer
mix docs      # must be warning-free
```

`mix lint` is exactly what CI gates on, so a local run and a CI run cannot disagree.

## Architectural rules

These are load-bearing. Changing them changes what MLServe is.

1. **Nothing goes on the prediction hot path that is not already there.** `predict/3` costs one
   ETS read, an atomic counter, and either a direct backend call or one `GenServer.call`. Adding a
   `GenServer.call` to a shared process would serialise every prediction on the node. If a feature
   seems to need one, it belongs in the caller or at load time.

2. **The registry is never called on a read.** `MLServe.ModelRegistry` owns the catalog ETS table
   and serialises writes. Reads are `:ets.lookup/2` in the calling process. Never add a
   `GenServer.call` to a read path.

3. **`MLServe.Route` stays slim.** It is copied out of ETS on every prediction. Only atoms, small
   integers and references. Anything that grows with model size belongs on `MLServe.ModelSpec`,
   which is read only for `model_status/2`.

4. **Hooks, cache and validation run in the caller, before dispatch.** A worker slot is the scarce
   resource. Never move work into a worker that does not need the model.

5. **Backend failures are surfaced, never swallowed.** `MLServe.Backend` catches only to attach the
   model, version, callback and stacktrace before returning a `MLServe.BackendError`. Do not add a
   `rescue` that returns a bare atom.

6. **The core has one runtime dependency.** `:telemetry`. New integrations are guides, optional
   dependencies guarded by `Code.ensure_loaded?/1`, or separate packages. Do not add a dependency
   for convenience.

7. **No ML runtime in the test suite.** Tests run against `test/support/backends.ex`. If a test
   needs a real model file or a Rust toolchain, the test is wrong.

## Testing rules

- **Every test gets a unique model name.** Use `load!/2` or `unique_name/1` from `MLServe.Case`.
  MLServe is a singleton application with a global catalog; unique names are what let the suite run
  `async: true` and actually exercise contention.
- **Never `async: false`.** A concurrency library whose test suite is serial is not testing the
  thing that matters.
- **Telemetry assertions must match on the model name.** Handlers are global — your handler
  receives other tests' events, and the `ref` does not isolate them. Use `assert_telemetry/4`.
- **Prefer `eventually/2` over `Process.sleep/1`.** A fixed sleep either flakes on slow CI or wastes
  time on fast machines.
- **New behaviour needs a test that fails without it.** Concurrency, drain and canary code in
  particular: several bugs in this codebase were only found because a test asserted the *count* of
  backend invocations, not just the result.
- **Run `mix test --repeat-until-failure 20` before pushing anything touching the batcher,
  the drain path, or worker selection.** Races here surface one run in twenty.

## Documentation rules

- Every public function needs a `@doc` with `## Examples`. Verify example output by running it,
  never by reasoning about it.
- Every module needs a `@moduledoc` explaining **why** it is shaped the way it is, not just what it
  does. The interesting content in this codebase is the reasoning.
- Doctests are executed contract. If an example is not runnable, do not write it as a doctest.
- A new option means updating: the moduledoc table in `MLServe.Config`, `usage-rules.md`, the
  relevant guide, and `CHANGELOG.md`.

## Adding an option

Options are validated centrally. To add one:

1. Add the key to `@model_keys` in `lib/ml_serve/config.ex` — unknown keys are rejected, so
   forgetting this makes the option unusable.
2. Add a field to `MLServe.ModelSpec`, and to `MLServe.Route` only if the hot path needs it.
3. Add a `fetch_*` clause with a validation error that names the model and says what is valid.
4. Surface it in `MLServe.model_status/2` if it is operationally interesting.
5. Test the valid case, the invalid case, and the default.

## Releasing

1. Update `@version` in `mix.exs`.
2. Add a `## [x.y.z] - YYYY-MM-DD` section to `CHANGELOG.md`. The publish workflow `awk`s these
   headings into the GitHub release notes, so the format is load-bearing.
3. Commit, tag `vx.y.z`, push the tag. `.github/workflows/publish.yml` verifies the tag matches
   `@version`, runs the full suite, publishes to Hex and creates the release.

Never publish by hand — the tag-versus-version guard exists because that mismatch is easy to make
and impossible to undo on Hex.
