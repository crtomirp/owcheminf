## 0.2.7 - Descriptor Pre-selector dashboard UI

- Fixed the macOS layout issue where the control panel could occupy most of the widget window.
- Forced a compact fixed-width control panel and expandable report area for a reliable 30:70 working layout.
- Reworked the overview tab into a richer dashboard-style report with KPI cards, filtering cascade, quality flags, cluster summary, data summary, and QSAR next-step guidance.


## 0.2.6 - Descriptor Pre-selector layout correction

- Corrected Descriptor Pre-selector split logic to enforce a true controls/report 30:70 layout.
- Splitter sizing now detects the actual Qt child order instead of assuming index order.


## 0.2.5 - Descriptor Pre-selector UI/report polish

- Improved Descriptor Pre-selector layout preference to approximately 30:70 left/right panel ratio.
- Replaced the plain-text summary with a publication/workflow-oriented HTML report.
- Added quality flags, filter cascade table, settings summary, correlation-cluster overview and downstream QSAR recommendation.

## 0.2.4

- Renamed Descriptor Explorer widget display name to **QSAR Descriptor Explorer** for easier discovery in Orange.
- Widget remains registered under **Cheminf - QSAR**.

# Changelog

## 0.2.3 - QSAR report/model hub compatibility

- Added a `Model Summary` input to `QSAR Report Generator`.
- `QSAR/QSPR Model Hub → Model Summary` can now be connected directly to the report generator.
- If both `Validation Summary` and `Model Summary` are connected, `Validation Summary` is used first.
- Updated QSAR report documentation with the recommended connection map.


All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.2] - 2026-05-17

### Added
- Added `Descriptor Explorer` widget to the public QSAR workflow palette.
- Added reusable `qsar_descriptor_explorer_service` for descriptor category inference, missing-value checks, low-variance flags, high-correlation redundancy detection, filtered descriptor output, and HTML/Markdown quality reports.
- Added tests for descriptor quality flags, category inference, and redundancy filtering.

### Changed
- Extended the recommended QSAR workflow to include descriptor matrix exploration before descriptor preselection and model training.

## [0.2.1] - 2026-05-17

### Added
- Added GitHub release automation workflow to build distributions and attach `sdist` and `wheel` assets to version-tagged releases.
- Added PyPI publishing workflow scaffold with `TestPyPI` and `PyPI` targets for future trusted-publishing based releases.
- Added release documentation under `docs/`, including a release process guide, a `v0.2.1` plan, and a concrete `v0.2.1` runbook.
- Added `scripts/prepare_release.sh` to update `pyproject.toml` and insert a changelog skeleton for the next release.
- Added `scripts/release_tag.sh` support for release preflight validation without creating or pushing a tag.

### Changed
- Updated repository README and package documentation to surface release badges, release docs, and the new release helper workflow.
- Tightened release preparation around the `owcheminf` GitHub repository so future patch releases are reproducible and less manual.

### Fixed
- Hardened `scripts/release_tag.sh` so version parsing no longer depends on `tomllib` availability in the system `python3`.

## [0.2.0] - 2026-05-17
### Curated palette update
- Moved `Mol Ketcher` and `Compound Detail Card` into `Cheminf - Core`.
- Moved `Pair Viewer` into `Cheminf - Search & Analysis`.
- Added `Cheminf - Filters & Alerts` as a dedicated public category for rule-based triage tools.


### Added
- Added `RDKit Descriptors` widget with categorized descriptor presets inspired by the existing Mordred/PaDEL descriptor workflow.


### Changed
- Moved `Mol Editor` from `Cheminf - Development` to `Cheminf - Core` so structure drawing/editing is part of the main workflow palette.
- Reorganized Orange widget discovery into five clearer user-facing categories: `Cheminf - Core`, `Cheminf - Search & Analysis`, `Cheminf - QSAR`, `Cheminf - Reactions`, and `Cheminf - Development`.
- Reduced QSAR palette clutter by keeping the main public workflow focused on dataset building, descriptor filtering, model hub, applicability domain, model explanation, reporting, and prediction packaging.
- Moved overlapping, legacy, diagnostic, optional-dependency-heavy, and experimental widgets to `Cheminf - Development` instead of removing them.
- Bumped package version to `0.2.0` for the GitHub-ready curated-layout release.

## [0.1.1] - 2025-04-29

### Added
- Packaging smoke tests: source-tree, wheel archive and install-from-wheel verification
- `Compound Detail Card` widget with FAIRMol-style inspection and search-profile outputs
- `PharmaFP Search` widget for fragment- and motif-guided library ranking
- `Matched Molecular Pairs`, `R-Group Decomposition`, `Scaffold Splitter` widgets
- Teaching datasets and 13 ready-made Orange workflows under `data/`
- CI pipeline via GitHub Actions with full test suite and wheel verification

### Changed
- Moved toward thinner widgets backed by a reusable `chemcore` service layer
- `Ketcher` editor now uses lazy WebEngine initialization to reduce macOS startup crashes

## [0.1.0] - 2025-02-02

### Added
- Initial release with core chemoinformatics widget set
- RDKit-based standardization, fingerprints, similarity, scaffold and descriptor widgets
- ChEMBL browser and bioactivity retriever
- QSAR regression and applicability domain widgets
- Reaction tools: `RDKit Reactor`, `Reaction Enumerator`, `Reaction Viewer`
- Embedded structure editors: `Mol Editor`, `Mol Ketcher`
- MIT license