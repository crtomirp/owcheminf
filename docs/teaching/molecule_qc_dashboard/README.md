# Teaching Package: Molecule QC Dashboard

This teaching package introduces molecule-level quality control as the first step of professional cheminformatics workflows.

## Main widget

```text
Cheminf - Processing → Molecule QC Dashboard
```

## Why this matters

Most QSAR, descriptor, fingerprint, ChEMBL, and docking workflows fail quietly when the input chemistry is not inspected first. Molecule QC teaches students that data quality is a chemical problem, not only a technical problem.

## Core workflow

```text
File → Molecule QC Dashboard → QC Report → Data Table
                         ├── Clean Data → Mol Standardizer
                         └── Problem Data → Data Table
```

## Outputs

- **Clean Data**: records without QC issues.
- **Problem Data**: records requiring review.
- **QC Report**: one row per molecule with issue codes and descriptors.
- **QC Summary**: aggregated issue counts.
- **Clean Molecules** and **Problem Molecules**: ChemMol lists for advanced workflows.

## Suggested lesson sequence

1. Load `examples/molecule_qc_dashboard/molecule_qc_training_set.csv`.
2. Run Molecule QC Dashboard with default settings.
3. Inspect `QC Summary`.
4. Inspect `QC Report` and identify issue codes.
5. Send `Clean Data` to `Mol Standardizer`.
6. Discuss which warnings should be fixed, tolerated, or domain-dependent.

## CLI use without Orange

```bash
owcheminf-molecule-qc \
  examples/molecule_qc_dashboard/molecule_qc_training_set.csv \
  --smiles-column smiles \
  --out-prefix outputs/molecule_qc_demo
```

This creates:

- `molecule_qc_demo.qc_report.csv`
- `molecule_qc_demo.clean.csv`
- `molecule_qc_demo.problems.csv`
- `molecule_qc_demo.summary.csv`
- `molecule_qc_demo.summary.json`
