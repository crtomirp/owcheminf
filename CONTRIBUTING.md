# Contributing

Contributions are welcome. The project is a conda-first Orange3 add-on, so please set up the development environment as described below before submitting changes.

## Development setup

```bash
conda env create -f environment.yml
conda activate owcheminf
pip install -e .[dev]
```

## Before submitting a pull request

Run the full test suite:

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

Build and verify the wheel:

```bash
python -m build
python -m unittest discover -s tests -p 'test_*smoke.py' -v
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

Open an issue at [github.com/crtomirp/chem-inf-widgets/issues](https://github.com/crtomirp/chem-inf-widgets/issues) and include the Orange version, Python version, and a minimal reproducible example.
