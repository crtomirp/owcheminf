# Release Process

This document describes the practical release flow for `chem-inf-widgets` in the `owcheminf` GitHub repository.

## Release Types

- patch: bugfixes, packaging fixes, documentation updates, workflow hardening
- minor: new widgets, meaningful UX changes, expanded workflows, non-breaking API additions
- major: breaking workflow, packaging, or service-layer changes

## Recommended Flow

1. update `CHANGELOG.md`
2. update version in `pyproject.toml`
3. run local checks
4. commit release prep
5. create annotated tag
6. push `main` and the tag
7. publish a GitHub release
8. publish to PyPI when ready

## Local Release Checks

Use the project environment:

```bash
conda activate owcheminf
python -m pip install -e .[dev]
python -m compileall -q src tests
python -m pytest -q
python -m build
python -m unittest discover -s tests -p 'test_*smoke.py' -v
python -m pytest tests/test_widget_smoke_tester.py tests/test_widget_import_smoke.py -q
```

Optional cleanup after local build:

```bash
owcheminf-clean-repo
```

## GitHub Automation

The repository includes two release-oriented workflows:

- `.github/workflows/release.yml`
  Builds `sdist` and `wheel` artifacts on version tags and uploads them to the GitHub release.

- `.github/workflows/publish-pypi.yml`
  Prepares trusted-publishing-based PyPI release publishing for future tags or manual dispatch.

## PyPI Notes

Before first real PyPI publication:

1. verify the package name you want to publish
2. configure PyPI trusted publishing for this repository
3. optionally configure TestPyPI first
4. confirm README rendering and package metadata

## Release Notes Source

Start from `CHANGELOG.md`, then compress into:

- 3-5 highlights
- important behavior changes
- any optional dependency caveats
- upgrade notes if relevant
