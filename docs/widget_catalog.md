# Widget catalog and release layout

This document describes the curated widget palette introduced in version `0.2.0`.
The goal is to make the add-on easier to understand for new users while keeping
specialized and experimental widgets available for developers and advanced users.

## Cheminf - Core

Entry-point widgets for most workflows:

- Molecule Import Hub
- Molecule Export Hub
- SDF Reader
- SDF Writer
- ChEMBL Browser
- ChEMBL Data Retriever
- Molecule QC Dashboard
- Mol Standardizer
- Mol Editor
- Mol Ketcher
- Compound Detail Card
- Molecular Viewer
- 3D Molecular Viewer

## Cheminf - Search & Analysis

Broadly useful cheminformatics tools:

- Substructure Search
- Similarity Search
- Fingerprint Generator
- RDKit Descriptors
- Mol Descriptors
- Scaffold Analysis
- Scaffold Splitter
- Diversity Picker
- Activity Cliff Finder
- R-Group Decomposition
- Matched Molecular Pairs
- Pair Viewer

## Cheminf - Filters & Alerts

Rule-based filtering and alert workflows:

- Drug Filter

## Cheminf - QSAR

Public QSAR/QSPR workflow and diagnostics:

- QSAR Dataset Builder
- QSAR Descriptor Explorer
- Descriptor Pre-selector
- QSAR/QSPR Model Hub
- QSAR Validation Dashboard
- Applicability Domain
- Model Explanation
- QSAR Report Generator
- QSAR Prediction Packager

## Cheminf - Reactions

Reaction-specific workflows:

- Reaction Viewer
- RDKit Reactor
- Reaction Enumerator

## Cheminf - Development

Widgets kept available but moved out of the main palette because they are
experimental, diagnostic, overlapping, specialized, or optional-dependency-heavy:

- Widget Smoke Tester
- Audit Trail Viewer
- PharmaFP Search
- Cyclic Registry Fingerprint
- ISIDA Descriptors
- PaDEL Descriptors
- Applicability Domain Workbench
- Atom Contribution Map
- QSAR Regression
- MLR Model Selection

## Suggested future public widgets

The next public-facing additions that would strengthen the package are:

- Molecular Space Map: PCA/UMAP/t-SNE projection of fingerprints or descriptors.
- Dataset Profiler: quick quality and diversity summary of a molecular dataset.
- Library Clustering: Butina, scaffold, or descriptor-based clustering with representatives.
- Chemical Series Explorer: scaffold/R-group/SAR table for related compound series.
- ADMET Radar: visual drug-likeness and rule-based filtering summary.
- Train/Test Splitter for QSAR: random, scaffold, cluster, and time-based splitting.
- Outlier & Leverage Inspector: residual, leverage, applicability-domain, and suspicious-compound view.
- Similarity Network: graph view of molecular similarity relationships.
