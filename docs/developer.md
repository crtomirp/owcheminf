# Developer Notes

This project is moving toward a thinner-widget architecture.

## Preferred code split

Widget responsibilities:

- UI layout
- Orange inputs and outputs
- settings
- progress and status display
- invoking service-layer functions

Service responsibilities:

- RDKit logic
- SMILES and molecule conversion
- descriptor generation
- search logic
- scaffold analysis
- QSAR helpers
- table-building utilities

## Important folders

- [src/chem_inf_widgets/widgets](../src/chem_inf_widgets/widgets)
- [src/chem_inf_widgets/chemcore/services](../src/chem_inf_widgets/chemcore/services)
- [src/chem_inf_widgets/chemcore/data](../src/chem_inf_widgets/chemcore/data)
- [tests](../tests)

## Current engineering direction

The most useful standardization targets for future cleanup are:

- shared SMILES-column detection
- shared Orange `Table <-> ChemMol` conversion helpers
- RDKit-safe parsing and validation helpers
- common progress/cancel patterns for long-running widgets
- consistent packaging and resource-loading helpers

## Recommended checks

### Core checks

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

### Packaging checks

```bash
python -m unittest tests.test_packaging_smoke tests.test_wheel_resource_smoke tests.test_wheel_install_smoke -v
```

### Orange widget import smoke

```bash
python -m unittest discover -s tests -p 'test_widget_import_smoke.py' -v
```

## Packaging guidance

- keep runtime resources inside package-data
- rebuild the wheel after packaging edits
- verify both source-tree resources and installed-wheel resources

See also:

- [packaging.md](packaging.md)
- [troubleshooting.md](troubleshooting.md)
