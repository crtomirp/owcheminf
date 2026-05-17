# Worksheet 1: First Molecule QC Dashboard

## Context

You received a small molecular dataset and need to decide whether it can be used for descriptor calculation or QSAR modelling.

## Input data

```text
examples/molecule_qc_dashboard/molecule_qc_training_set.csv
```

## Orange workflow

```text
File → Molecule QC Dashboard → QC Summary → Data Table
                         ↓
                      QC Report → Data Table
```

## Tasks

1. Load the CSV file with the File widget.
2. Connect it to Molecule QC Dashboard.
3. Open QC Summary in a Data Table.
4. Open QC Report in a second Data Table.
5. Count clean, problem, invalid, and duplicate records.

## Questions

1. Which molecule records are invalid?
2. Which records are duplicates?
3. Which records have disconnected fragments?
4. Which records would you send directly to descriptor calculation?
5. Which records require chemical review?

## Expected product

A one-page QC note describing the dataset quality and the recommended next action.
