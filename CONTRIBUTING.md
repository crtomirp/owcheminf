# Contributing

Contributions are welcome. The project is a conda-first Orange3 add-on, so please set up the development environment as described below before submitting changes.

## Development setup

```bash
conda env create -f environment.yml
conda activate owcheminf
python -m pip install -e .[dev]
```

## Before submitting a pull request

Run the main local test suite:

```bash
python -m compileall -q src tests
python -m pytest -q
```

Run the packaging and widget smoke checks:

```bash
python -m build
python -m unittest discover -s tests -p 'test_*smoke.py' -v
python -m pytest tests/test_widget_smoke_tester.py tests/test_widget_import_smoke.py -q
```

Check code style:

```bash
ruff check src tests
black --check src tests
isort --check-only src tests
```

## Code layout

- `src/chem_inf_widgets/widgets/` — Orange widget classes (UI, signals, settings)
- `src/chem_inf_widgets/chemcore/services/` — reusable chemoinformatics logic
- `src/chem_inf_widgets/chemcore/data/` — packaged runtime resources (JSON, CSV)
- `tests/` — service unit tests, packaging smoke tests and widget import checks

The preferred pattern is a thin widget that delegates to a service function. New chemistry logic should go into `chemcore/services/`, not directly into the widget class.

## Reporting bugs

Open an issue at [github.com/crtomirp/owcheminf/issues](https://github.com/crtomirp/owcheminf/issues) and include the Orange version, Python version, the active install method, and a minimal reproducible example.

## Pull request notes

- Keep widget UI changes and service-layer chemistry logic separated when practical.
- Add or update tests for widget wiring, service behavior, or packaging whenever behavior changes.
- If a change depends on optional packages such as `mordred`, `optuna`, `py3Dmol`, or `torch`, mention that clearly in the PR description.
- Prefer small, reviewable commits over one very large mixed refactor.
