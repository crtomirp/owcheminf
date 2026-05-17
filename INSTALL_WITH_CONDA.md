# Install With Conda

This project works best from a `conda-forge` environment because `Orange3`, `RDKit`, `Qt` and `QWebEngine` are more reliable there than in a plain `pip` environment.

## Recommended setup

```bash
conda env create -f environment.yml
conda activate owcheminf
pip install -e .
```

That gives you the core widget set plus the Orange/Qt runtime used during development.

## Developer setup

If you want the broader local toolbox, optional scientific extras and packaging tools in one environment:

```bash
conda env create -f environment-dev.yml
conda activate owcheminf-dev
pip install -e .
```

## Optional features

Descriptor widgets:

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

Everything at once:

```bash
pip install -e .[all]
```

## Build and verify a wheel

```bash
python -m build
python -m unittest discover -s tests -p 'test_*smoke.py' -v
```

## Launch Orange

```bash
orange-canvas
```

If `Ketcher` or other WebEngine-based widgets are unstable on macOS, see [docs/troubleshooting.md](docs/troubleshooting.md).
