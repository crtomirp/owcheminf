# Optional Dependencies and Clean Install Checks

This project now supports a lean install path and a richer optional-feature path.

Use this guide when you want to:

- create a fresh runtime environment
- understand which widgets need optional packages
- verify that a clean install is healthy before manual testing

## Lean runtime install

This is the recommended baseline for everyday Orange usage and for CI-like validation.

```bash
conda env create -f environment.yml
conda activate owcheminf
python -m pip install -e .
```

If you want to run the pytest-based smoke checks from the lean environment, add:

```bash
python -m pip install pytest
```

## Developer install

Use this when you want broader tooling, packaging helpers, and common optional extras already present.

```bash
conda env create -f environment-dev.yml
conda activate owcheminf-dev
python -m pip install -e .
```

## Clean install validation

After a fresh install, run these checks:

```bash
python -m unittest discover -s tests -p 'test_packaging_smoke.py' -v
python -m unittest discover -s tests -p 'test_widget_import_smoke.py' -v
python -m pytest tests/test_widget_smoke_tester.py tests/test_audit_trail_viewer.py tests/test_workflow_provenance_e2e.py -q
```

For a manual Orange-side sanity check, open:

- `Widget Smoke Tester`
- `Audit Trail Viewer`

Then run:

1. `All Widgets` with `Instantiate widgets` disabled
2. `Core Workflow Smoke`
3. `Workflow Suite`

## Optional feature matrix

Core widgets should still import without these extras. Missing extras may disable a feature, reduce functionality, or show a runtime message instead of crashing.

| Extra / package | Main widgets affected | Behavior when missing |
| --- | --- | --- |
| `optuna` / `.[hpo]` | `QSAR/QSPR Model Hub` | Widget still imports; HPO and `auto` model are disabled. |
| `mordred` / `.[descriptors]` | `Mol Descriptor` | Widget remains available but descriptor computation is disabled with a clear message. |
| `padelpy` and Java | `PaDEL Descriptors` | PaDEL-specific computation fails until Java and `padelpy` are available. |
| `py3Dmol` / `.[viewer3d]` | `3D Molecular Viewer` | 3D rendering features are unavailable. |
| `torch` / `.[deep-learning]` | future or optional DL workflows | Deep-learning features remain unavailable. |

## Clean install troubleshooting

If a fresh environment behaves differently from an existing development environment:

1. confirm the active interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

2. confirm the package path:

```bash
python -c "import chem_inf_widgets; print(chem_inf_widgets.__file__)"
```

3. rerun the smoke checks above before manual widget debugging
4. if only one optional widget is affected, compare installed extras rather than rebuilding the whole environment first

## Notes for maintainers

- `environment.yml` is the lean baseline and should stay usable on its own
- `environment-dev.yml` can include broader tooling and convenience extras
- widget imports should degrade gracefully when optional packages are missing
- `Widget Smoke Tester` and `Audit Trail Viewer` are intended to be the first-line runtime diagnostics for contributors
