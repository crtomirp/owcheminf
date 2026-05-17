# Troubleshooting

## Orange starts but a widget is missing

Try these steps:

1. reinstall the package from the current source tree or fresh wheel
2. fully restart `orange-canvas`
3. verify that Orange is using the same Python environment where the package is installed

Useful check:

```bash
python -c "import chem_inf_widgets; print(chem_inf_widgets.__file__)"
```

## Ketcher or WebEngine widgets crash on macOS

`Qt WebEngine` can be sensitive on macOS, especially inside Conda environments.

Try:

```bash
QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu" orange-canvas
```

If that is still unstable:

```bash
QTWEBENGINE_DISABLE_SANDBOX=1 orange-canvas
```

The `Ketcher` widget in this project uses lazy WebEngine initialization to reduce startup crashes, but system-level Qt/Chromium issues can still surface.

## PaDEL descriptors do not run

Check that Java is available:

```bash
java -version
```

Also verify `padelpy`:

```bash
python -c "import padelpy; print(padelpy.__file__)"
```

If needed:

```bash
pip install padelpy
```

## Mordred widget is disabled

Install the descriptor extra:

```bash
pip install -e .[descriptors]
```

or directly:

```bash
pip install mordred
```

## A wheel installs, but resources are missing at runtime

Rebuild and re-test the wheel:

```bash
python -m build
python -m unittest tests.test_packaging_smoke tests.test_wheel_resource_smoke tests.test_wheel_install_smoke -v
```

If the wheel archive is missing files, inspect `pyproject.toml` package-data settings.
