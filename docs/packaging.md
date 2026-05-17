# Packaging Notes

This package uses a `src/` layout and ships widget assets directly inside the wheel.

## Build steps

```bash
python -m build
```

This produces:

- `dist/chem_inf_widgets-<version>.tar.gz`
- `dist/chem_inf_widgets-<version>-py3-none-any.whl`

## Resource expectations

The built wheel must contain runtime assets used by widgets and services, especially:

- `chemcore/data/smartspains.json`
- `chemcore/data/pharmafp250.json`
- `chemcore/data/cyclic_registry.json`
- `chemcore/data/patterns.csv`
- `chemcore/resources/padel_presets/*.xml`
- widget icons under `widgets/icons/`
- embedded editor assets under `widgets/jsme/` and `widgets/ketcher/`

## Verification

Run the packaging smoke tests after every wheel build:

```bash
python -m unittest discover -s tests -p 'test_*smoke.py' -v
```

The checks cover three layers:

1. source-tree resource presence
2. wheel archive contents
3. install-from-wheel resource access in a clean temporary virtual environment

## Distribution checklist

Before publishing or sharing a wheel:

1. remove stale `build/`, `dist/`, `*.egg-info/` artifacts
2. build a fresh wheel from the current source tree
3. run the packaging smoke tests
4. optionally verify Orange widget imports in the `owcheminf` environment

## Optional dependency policy

Core runtime dependencies live in `pyproject.toml`.

Conda environment files follow the same split:

- `environment.yml` for the lean runtime used to run Orange and the core widgets
- `environment-dev.yml` for local development, packaging tools and common optional extras

Optional features are split into extras:

- `descriptors`
- `viewer3d`
- `hpo`
- `deep-learning`
- `dev`
- `all`

This keeps a minimal install usable while allowing richer local development environments.
