# Worksheet 05 — CLI Batch Import Without Orange

## Context

For large datasets or reproducible workflows, import should also work without the Orange GUI.

## CLI command

```bash
mkdir -p outputs
owcheminf-molecule-import \
  examples/molecule_import_hub/molecule_import_demo.csv \
  --smiles-column smiles \
  --name-column name \
  --out-prefix outputs/import_demo \
  --json
```

## Output files

```text
outputs/import_demo.import_report.csv
outputs/import_demo.molecules.csv
outputs/import_demo.failed.csv
outputs/import_demo.summary.csv
outputs/import_demo.summary.json
```

## Tasks

1. Run the CLI command.
2. Open the generated CSV files.
3. Compare CLI output with Orange widget output.
4. Write a reproducibility note describing the command and input file.

## Questions

1. Why is CLI useful for batch processing?
2. Why should the command be included in a methods section?
3. What should be archived together with the imported dataset?
