# chem-inf-widgets

[![CI](https://github.com/crtomirp/owcheminf/actions/workflows/ci.yml/badge.svg)](https://github.com/crtomirp/owcheminf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

`chem-inf-widgets` is an Orange3 chemoinformatics add-on built around `RDKit`, `Orange`, `ChEMBL`, descriptor generation, QSAR analysis, scaffold analytics, similarity search and reaction workflows.

## Start Here

- main package README: [src/chem_inf_widgets/README.md](src/chem_inf_widgets/README.md)
- conda installation guide: [INSTALL_WITH_CONDA.md](INSTALL_WITH_CONDA.md)
- packaging notes: [docs/packaging.md](docs/packaging.md)
- optional dependency guide: [docs/optional_dependencies.md](docs/optional_dependencies.md)
- troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md)
- developer notes: [docs/developer.md](docs/developer.md)
- widget notes: [docs/widgets/README.md](docs/widgets/README.md)

## Repository Layout

- [src](src) contains the Orange add-on code
- [tests](tests) contains service, packaging and widget smoke tests
- [data](data) contains teaching datasets and Orange workflows
- [docs](docs) contains the current project documentation

## GitHub Workflow

- bug reports: [open a bug report](https://github.com/crtomirp/owcheminf/issues/new?template=bug_report.md)
- feature ideas: [open a feature request](https://github.com/crtomirp/owcheminf/issues/new?template=feature_request.md)
- pull requests: open against `main`
- contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- change log: [CHANGELOG.md](CHANGELOG.md)
- security policy: [SECURITY.md](SECURITY.md)

## Quick Setup

```bash
conda env create -f environment.yml
conda activate owcheminf
pip install -e .
orange-canvas
```

For a fuller local developer setup with optional tooling already present:

```bash
conda env create -f environment-dev.yml
conda activate owcheminf-dev
pip install -e .
orange-canvas
```

For optional features such as `Mordred`, `PaDEL`, `py3Dmol`, `optuna` or `torch`, use the extras documented in [src/chem_inf_widgets/README.md](src/chem_inf_widgets/README.md) and the clean-install notes in [docs/optional_dependencies.md](docs/optional_dependencies.md).

## Packaging Smoke Tests

```bash
python -m build
python -m unittest discover -s tests -p 'test_*smoke.py' -v
```

## Runtime Smoke Checks

```bash
python -m pip install pytest
python -m unittest discover -s tests -p 'test_widget_import_smoke.py' -v
python -m pytest tests/test_widget_smoke_tester.py tests/test_audit_trail_viewer.py tests/test_workflow_provenance_e2e.py -q
```

## Local Cleanup

```bash
owcheminf-clean-repo --dry-run
owcheminf-clean-repo
```

This removes local build/cache artifacts such as `build/`, `dist/`, source-side `__pycache__`, `*.pyc`, and generated `.setuptools-egg-info/*.egg-info` metadata while keeping the tracked placeholder directory intact.
