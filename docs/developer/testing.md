# Testing Guide

This package uses a layered testing strategy. The fastest and most stable tests are pure service tests. Widget tests are added when behavior depends on Orange signals, outputs, or UI state.

## Test categories

### Pure service tests

Use these when the logic does not need Qt widgets.

Typical targets:

- chemistry helpers
- descriptor generators
- QSAR utilities
- splitters
- profiling/report builders

Good traits:

- fast
- deterministic
- easy to isolate edge cases

### Orange table contract tests

Use these when the service depends on Orange `Table`, `Domain`, or variable roles but not on widget UI.

Typical targets:

- `Table -> ChemMol` conversion
- role inference
- Orange summary/report table builders

### Widget tests

Use these when you need to confirm:

- input handlers call the right logic
- outputs are sent correctly
- warning messages are shown or cleared
- a widget reacts correctly to partial failures

### Smoke tests

Use smoke tests to catch:

- import failures
- registration failures
- workflow-level regressions
- packaging and wheel issues

## Service testing without Qt

Prefer `pytest` with small synthetic inputs.

Typical pattern:

```python
result = run_service(input_data, config)
assert result.issues == []
assert result.summary["something"] == expected
```

When possible:

- keep test matrices tiny
- use hand-built SMILES examples
- assert exact issue codes, not only string fragments

## Widget testing with Orange

Widget tests usually need:

- `pytest.importorskip("Orange")`
- `pytest.importorskip("AnyQt")`
- a `QApplication` instance
- `patch.object(..., "send", ...)` on outputs when needed

Typical checks:

- warning shown after backend issue
- warning cleared after clean rerun
- output becomes `None` when input disappears
- status text updates

## Optional dependency policy in tests

Do not make the whole suite depend on optional packages.

Rules:

- guard tests with `pytest.importorskip(...)` when the package is optional
- test graceful fallback behavior when the package is missing
- do not silently skip required-runtime dependency failures

Examples:

- `mordred`
- `padelpy`
- `torch`
- `umap-learn`

For optional fallbacks, prefer explicit tests like:

- “UMAP missing falls back to PCA”
- “descriptor backend unavailable reports warning”

## What to test for new services

For a new service, aim for:

- one happy-path test
- one failure-path test
- one edge-case test

Examples:

- valid matrix returns coordinates
- empty input returns error issue
- optional dependency missing triggers fallback

## What to test for widget changes

For widget refactors, focus on behavior:

- outputs unchanged
- warnings still appear
- warnings clear correctly
- no input path still behaves correctly

Avoid brittle tests that inspect large chunks of UI layout unless the layout itself is the feature.

## Useful commands

### Fast targeted checks

```bash
python -m pytest -q tests/test_dataset_profiler_service.py
python -m pytest -q tests/test_dataset_profiler_widget.py
python -m compileall src
```

### Wider checks

```bash
python -m pytest -q tests/test_qsar*
python -m pytest -q tests/test_widget_smoke_tester.py
python -m pytest tests -q
```

### Packaging checks

```bash
python -m build
python -m pytest tests/test_packaging_smoke.py tests/test_wheel_resource_smoke.py tests/test_wheel_install_smoke.py -q
```

## Clean test-writing conventions

- keep fixtures small and local
- prefer explicit names over clever helpers
- avoid broad monkeypatching when a narrow patch is enough
- assert structured fields like `issue.code`
- add tests only for behavior that changed

## Suggested review checklist

Before finishing a change, ask:

- did I add a test for the new behavior?
- does the test fail without the change?
- is the test using the thinnest possible layer?
- does it cover both success and failure if the code now reports issues?
