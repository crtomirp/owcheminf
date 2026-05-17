# Teaching Package: Molecule Import Hub

This mini-package introduces robust molecular data import as the first step in a professional cheminformatics workflow.

## Learning goals

Students will learn to:

1. distinguish table, SMILES and SDF-based molecular data sources;
2. identify structure, name and metadata columns;
3. inspect import failures instead of silently ignoring them;
4. connect import reporting to quality control and standardization;
5. use the same import logic through Orange or CLI.

## Main workflow

```text
Molecule Import Hub → Molecule QC Dashboard → Mol Standardizer → Fingerprint/QSAR widgets
```

## Worksheets

- `worksheet_01_importing_csv_smiles.md`
- `worksheet_02_importing_smi_files.md`
- `worksheet_03_import_report_and_failed_records.md`
- `worksheet_04_import_to_qc_pipeline.md`
- `worksheet_05_cli_batch_import.md`
