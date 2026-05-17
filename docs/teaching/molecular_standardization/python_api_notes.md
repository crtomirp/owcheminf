# Non-Orange Use: Python API Notes

This teaching package is primarily designed for Orange, but the standardization service can also be used from Python.

> Note: this is intended for teaching and scripting. For full batch CLI standardization, a dedicated command-line tool can be added later.

## Minimal Python example

```python
from chem_inf_widgets.chemcore.services.mol_standardizer import MolStandardizer, StandardizeConfig

standardizer = MolStandardizer(StandardizeConfig(
    cleanup=True,
    normalize=True,
    metal_disconnect=True,
    largest_fragment=True,
    reionize=True,
    uncharge=True,
    sanitize_before=True,
    sanitize_after=True,
    canonical_smiles=True,
))

for smi in ["[Na+].[O-]C(=O)C", "O=[N+]([O-])c1ccccc1"]:
    result = standardizer.standardize_smiles(smi)
    print(smi, "→", result.output_smiles, result.ok, result.log)
```

## Why teach both Orange and Python?

Orange is useful for visual, interactive learning. Python is useful for reproducible scripts, pipelines, notebooks, and automated preprocessing before QSAR or docking analysis.

## Suggested exercise

Ask students to run the same molecules in Orange and in Python, then compare:

- original SMILES;
- standardized SMILES;
- standardization logs;
- failed molecules.
