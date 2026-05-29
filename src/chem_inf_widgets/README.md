# chem-inf-widgets

`chem-inf-widgets` is an Orange3 add-on for chemoinformatics workflows built around `RDKit`, `Orange`, `ChEMBL`, descriptor generation, QSAR analysis, scaffold analytics, similarity search and reaction-based exploration.

The project is designed for three main use cases:

- day-to-day chemoinformatics work inside Orange
- teaching workflows for medicinal chemistry and QSAR
- reusable service-layer code for notebooks and future FAIRMol-style tools

## Highlights

- Orange3 widget add-on with more than 30 chemoinformatics widgets
- `RDKit`-based standardization, filtering, fingerprints, similarity and scaffold logic
- descriptor support through native RDKit, `Mordred`, `PaDEL` and `ISIDA`
- ChEMBL retrieval and browser widgets
- QSAR, applicability domain and activity-cliff analysis
- reaction tools, structure viewers and embedded sketchers
- packaged teaching datasets and ready-made `.ows` workflows

## Installation

### Recommended: Conda

This package works best in a `conda-forge` environment because `Orange3`, `RDKit`, `Qt` and `QWebEngine` are much more reliable there than in a plain `pip` install.

```bash
conda env create -f environment.yml
conda activate owcheminf
pip install -e .
```

Detailed setup notes are in [INSTALL_WITH_CONDA.md](../../INSTALL_WITH_CONDA.md).
Clean-install and optional dependency notes are in [docs/optional_dependencies.md](../../docs/optional_dependencies.md).

For the fuller developer environment with optional tooling preinstalled:

```bash
conda env create -f environment-dev.yml
conda activate owcheminf-dev
pip install -e .
```

### Optional extras

Descriptor extras:

```bash
pip install -e .[descriptors]
```

3D viewer extras:

```bash
pip install -e .[viewer3d]
```

QSAR HPO extras:

```bash
pip install -e .[hpo]
```

Deep learning extras:

```bash
pip install -e .[deep-learning]
```

Everything:

```bash
pip install -e .[all]
```

If you want to understand which widgets degrade gracefully when an extra is missing, see [docs/optional_dependencies.md](../../docs/optional_dependencies.md).

### Launch Orange

```bash
orange-canvas
```

## Widget Overview

### Input / Output

- `Molecule Import Hub`
- `Molecule Export Hub`
- `SDF Reader`
- `SDF Writer`

### Editors / Viewers

- `Mol Editor`
- `Mol Ketcher`
- `Molecular Viewer`
- `3D Molecular Viewer`
- `Pair Viewer`
- `Compound Detail Card`

### Standardization / Filtering / Search

- `Audit Trail Viewer`
- `Mol Standardizer`
- `Drug Filter`
- `Substructure Search`
- `Similarity Search`
- `PharmaFP Search`

### Descriptors / Fingerprints

- `Fingerprint Generator`
- `Mol Descriptor`
- `PaDEL Descriptors`
- `ISIDA Descriptors`

### Analysis

- `Scaffold Analysis`
- `Scaffold Splitter`
- `Diversity Picker`
- `Applicability Domain`
- `Activity Cliff Finder`
- `Matched Molecular Pairs`
- `R-Group Decomposition`

### Data Retrieval

- `ChEMBL Browser`
- `ChEMBL Bioactivity Retriever`

### Modeling

- `QSAR Dataset Builder`
- `QSAR Descriptor Explorer`
- `Descriptor Pre-selector`
- `QSAR/QSPR Model Hub`
- `QSAR Validation Dashboard`
- `Applicability Domain`
- `Model Explanation`
- `QSAR Report Generator`
- `QSAR Prediction Packager`
- `QSAR Regression`
- `MLR Model Selection`

### Reactions

- `RDKit Reactor`
- `Reaction Enumerator`
- `Reaction Viewer`

## Teaching Material

The repository includes curated teaching assets:

- datasets in [data/7_teaching_workflows](../../data/7_teaching_workflows)
- `15` Orange workflows in [data/ows_workflow](../../data/ows_workflow)

These cover:

- standardization and filtering
- diversity selection
- activity cliffs
- QSAR regression
- applicability domain
- chemical space maps
- similarity clustering
- outlier hunting
- activity landscape exploration
- compound-detail driven database search

## Packaging and Distribution

To build a wheel:

```bash
python -m build
```

To verify packaged resources:

```bash
python -m unittest discover -s tests -p 'test_*smoke.py' -v
```

Additional notes are in:

- [docs/packaging.md](../../docs/packaging.md)
- [docs/release_process.md](../../docs/release_process.md)
- [docs/troubleshooting.md](../../docs/troubleshooting.md)

## Development Notes

The codebase is split into:

- [src/chem_inf_widgets/widgets](widgets) for Orange widgets
- [src/chem_inf_widgets/chemcore/services](chemcore/services) for reusable chemoinformatics logic
- [src/chem_inf_widgets/chemcore/data](chemcore/data) for packaged runtime resources
- [tests](../../tests) for service, packaging and widget smoke tests

The project is intentionally moving toward thinner widgets and a stronger service layer, so the same logic can be reused in Orange, notebooks and future web tooling.

## Troubleshooting

If a widget does not appear in Orange:

1. reinstall the package from the current source tree or fresh wheel
2. fully restart `orange-canvas`
3. confirm Orange is using the same Python environment where `chem-inf-widgets` is installed

If `Ketcher` or other WebEngine-based widgets are unstable on macOS, see [docs/troubleshooting.md](../../docs/troubleshooting.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
