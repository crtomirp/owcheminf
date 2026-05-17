# Molecule Import Hub Examples

This folder contains small files for testing and teaching the **Molecule Import Hub**.

## Files

- `molecule_import_demo.csv` — CSV input with valid, salt-like, aromatic/kekulized and invalid SMILES examples.
- `mini_import.smi` — simple whitespace-delimited SMILES file.

## Orange workflow

```text
Molecule Import Hub → Molecule QC Dashboard → Mol Standardizer → Cyclic Registry Fingerprint
```

## CLI example

```bash
owcheminf-molecule-import \
  examples/molecule_import_hub/molecule_import_demo.csv \
  --smiles-column smiles \
  --name-column name \
  --out-prefix outputs/import_demo \
  --json
```
