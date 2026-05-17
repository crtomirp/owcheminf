# Molecular Standardization Example Data

## Files

- `standardization_training_set.csv` — mixed teaching dataset containing salts, heterocycles, aromatic/kekulized pairs, invalid inputs, mixtures, and drug-like molecules.
- `standardization_edge_cases.csv` — smaller dataset focused on difficult or chemically ambiguous cases.

## Recommended Orange workflow

```text
File → Mol Standardizer → Data Table
```

Then connect the standardized output to:

```text
Mol Standardizer → Fingerprint Generator
Mol Standardizer → Cyclic Registry Fingerprint
Mol Standardizer → Mol Descriptors 2
```

## Important columns

- `name`: molecule identifier
- `smiles`: input structure
- `case_type`: teaching category
- `comment` or `expected_issue`: interpretation hint
