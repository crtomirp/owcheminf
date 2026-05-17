# Instructor Guide: Molecule QC Dashboard

## Learning goals

Students should be able to:

1. explain why molecular QC is required before standardization, descriptors, fingerprints, QSAR, and docking;
2. identify common structure problems such as invalid SMILES, salts, duplicates, metals, charges, and unspecified stereochemistry;
3. distinguish between fatal errors and domain-dependent warnings;
4. design a simple QC decision policy for a modelling workflow.

## Recommended duration

- Short version: 45 minutes
- Full practical: 90 minutes
- Extended project: 2–3 hours

## Instructor preparation

Before class, verify:

```bash
conda activate owcheminf
pip install -e .
python -c "import chem_inf_widgets.widgets.ow_molecule_qc_dashboard"
owcheminf-molecule-qc --help
```

## Classroom discussion prompts

- Should charged molecules always be removed?
- Should metal-containing compounds be removed from a drug-like QSAR dataset?
- Is a duplicate always an error?
- Why can unspecified stereochemistry be a serious modelling problem?
- Which QC decisions depend on the project domain?

## Assessment evidence

Students should submit:

- a QC report table,
- a short explanation of the top three issue codes,
- a clean/problem split,
- a short QC policy statement.
