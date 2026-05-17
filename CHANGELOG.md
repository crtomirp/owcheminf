# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
