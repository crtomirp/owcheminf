# Instructor Guide: Molecule Import Hub

## Suggested duration

60–90 minutes.

## Required preparation

Install the package and confirm that the widget appears in Orange:

```bash
pip install -e .
orange-canvas
```

Then locate:

```text
Cheminf - Data → Molecule Import Hub
```

## Teaching emphasis

Do not treat molecular import as a trivial file-opening step. In real projects, import decisions affect every downstream analysis: standardization, duplicate detection, descriptors, fingerprints, QSAR, applicability domain and reports.

## Assessment evidence

Ask students to submit:

- a screenshot or exported import report;
- a list of failed records and reasons;
- a short explanation of how they selected the SMILES/name columns;
- a downstream QC summary generated from imported molecules.
