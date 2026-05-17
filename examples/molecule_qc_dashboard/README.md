# Molecule QC Dashboard example data

This folder contains small teaching datasets for the Phase 3 **Molecule QC Dashboard**.

## Files

- `molecule_qc_training_set.csv` — mixed clean/problematic records for Orange.
- `mini_qc.smi` — compact SMILES input for the CLI.

## Orange workflow

```text
File → Molecule QC Dashboard → QC Report → Data Table
                         ├── Clean Data → Mol Standardizer
                         └── Problem Data → Data Table
```

## CLI example

```bash
owcheminf-molecule-qc \
  examples/molecule_qc_dashboard/molecule_qc_training_set.csv \
  --smiles-column smiles \
  --out-prefix outputs/molecule_qc_demo
```
