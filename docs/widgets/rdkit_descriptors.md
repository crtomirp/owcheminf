# RDKit Descriptors

## Status

Current widget in the package. Uses RDKit directly; no optional descriptor backend is required beyond the package's standard `rdkit` dependency.

Source:
- [ow_rdkit_descriptors.py](../../src/chem_inf_widgets/widgets/ow_rdkit_descriptors.py)
- [rdkit_descriptor_service.py](../../src/chem_inf_widgets/chemcore/services/rdkit_descriptor_service.py)

## Purpose

`RDKit Descriptors` calculates numerical molecular descriptors from an Orange `Table` containing SMILES strings or from a list of `ChemMol` objects. The descriptors are appended as continuous Orange variables, so the output can be connected directly to widgets such as `Data Table`, `Select Columns`, `PCA`, `Rank`, `Random Forest`, `QSAR Model Hub`, or `Applicability Domain`.

Use this widget when you need a fast, transparent descriptor set for teaching, chemical-space exploration, QSAR/QSPR modelling, or first-pass molecular dataset inspection.

## Input

The widget accepts either of the following inputs:

- `Data` — Orange `Table`; the widget searches for a string column named `SMILES` or containing `smiles` in the name.
- `Molecules` — a list of `ChemMol` objects from other OWChemInf widgets.

If both inputs are present, table input is preferred when it contains valid SMILES. The `Molecules` input is used as a fallback when no valid table molecules are available.

## Output

- `Data` — Orange `Table` with selected RDKit descriptor columns appended as continuous variables.
- `Molecules` — original `ChemMol` list. When enabled, descriptor values are also written into the molecule properties.

Rows with invalid or empty SMILES are preserved in the output table, but descriptor values for those rows are set to missing values. The widget reports how many rows were skipped.

## Main controls

### Preset

Choose one of the descriptor presets:

- `Recommended RDKit QSAR core` — compact default set for QSAR/QSPR teaching and robust first-pass modelling.
- `Descriptor family: physicochemical / drug-like` — molecular weight, logP, TPSA, molar refractivity, QED, charge, and surface-area descriptors.
- `Descriptor family: constitutional and counts` — atom, heteroatom, ring, hydrogen-bond, and rotatable-bond counts.
- `Descriptor family: topology and connectivity` — Balaban, Bertz, Chi, Kappa, Hall-Kier, and related graph descriptors.
- `Descriptor family: VSA / BCUT / EState` — surface-area, BCUT, charge-related, and EState descriptors.
- `Descriptor family: fragments / functional groups` — RDKit fragment counters and Morgan fingerprint density descriptors.
- `Custom / manual category selection` — select descriptor categories manually and then add individual descriptors.
- `All RDKit descriptors` — expose the full descriptor catalogue available in the installed RDKit version.

### Descriptor categories

Visible in custom mode. Categories group descriptors into chemically meaningful families, for example physicochemical descriptors, counts, topology, VSA, BCUT, and fragments.

### Descriptor selection

Use the `Available` and `Selected` lists to build the final descriptor set. The selected list controls both which descriptors are calculated and the order in which they appear in the output table.

### Write descriptor values into Molecules

When enabled, calculated descriptor values are stored as properties on the outgoing `ChemMol` objects. This is useful when downstream widgets work primarily with molecule lists rather than Orange tables.

### Auto-run

When enabled, the widget recomputes descriptors automatically after a new input or setting change. For large datasets, keep this disabled and press `Compute` manually.

## Recommended workflows

### QSAR/QSPR dataset preparation

1. `Molecule Import Hub` or `SDF Reader`
2. `Mol Standardizer`
3. `RDKit Descriptors`
4. `Descriptor Filter` or `Descriptor Explorer`
5. `QSAR Model Hub`
6. `QSAR Validation Dashboard`

### Chemical-space exploration

1. `Molecule Import Hub`
2. `RDKit Descriptors`
3. `PCA`, `t-SNE`, or `Molecular Space Map`
4. `Scatter Plot`

### Teaching descriptor interpretation

1. Load a small table containing familiar molecules.
2. Select the `Recommended RDKit QSAR core` preset.
3. Inspect descriptors such as `MolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`, `RingCount`, and `FractionCSP3` in `Data Table`.
4. Discuss how descriptors encode molecular size, polarity, lipophilicity, hydrogen bonding, and topology.

## Notes and limitations

- Descriptor availability can vary slightly between RDKit versions.
- Descriptors are calculated from the RDKit molecule generated from SMILES or from the incoming `ChemMol` object.
- Most descriptors are 2D descriptors. They do not replace 3D conformational, quantum-chemical, or force-field descriptors.
- Missing descriptor values usually indicate an invalid input molecule or a descriptor calculation failure for a specific molecule.
- Standardize molecules before descriptor calculation when comparing compounds from mixed sources.

## Troubleshooting

### The widget says no valid molecules are available

Check that the input table contains a text column named `SMILES` or with `smiles` in the column name. Also check that the structures are valid RDKit-readable SMILES.

### Descriptor columns are missing values

Some rows probably contain invalid SMILES or molecules for which a specific descriptor failed. Inspect the warning message and validate the corresponding structures with `Molecule QC Dashboard` or `Mol Standardizer`.

### Too many descriptor columns are produced

Use `Recommended RDKit QSAR core` or one of the descriptor-family presets instead of `All RDKit descriptors`. You can also connect the output to `Descriptor Filter` or `Select Columns`.

## References

- RDKit `rdkit.Chem.Descriptors` documentation: <https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html>
- RDKit Python getting-started documentation, descriptor section: <https://www.rdkit.org/docs/GettingStartedInPython.html#calculating-all-descriptors>
