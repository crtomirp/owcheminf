# Teaching Materials

This directory contains teaching materials integrated into the `chem-inf-widgets` package.

| Module | Folder | Main focus |
|---|---|---|
| Cyclic Registry Fingerprint | `cyclic_registry_fingerprint/` | Interpretable heterocycle and cyclic motif fingerprinting. |
| QSAR Widgets | `qsar_widgets/` | Descriptor calculation, fingerprints, scaffold splitting, QSAR modelling, activity cliffs, and reporting. |
| Applicability Domain | `applicability_domain/` | QSAR prediction reliability, Williams leverage, kNN distance, Mahalanobis distance, and outlier triage. |
| Molecular Standardization | `molecular_standardization/` | Structure cleaning, salts, charges, aromaticity, standardization provenance, and QSAR-ready molecules. |
| ChEMBL Bioactivity Curation | `chembl_bioactivity_curation/` | ChEMBL target/bioactivity retrieval, assay filtering, duplicate handling, and QSAR-ready curation. |
| Scaffold and Activity Cliff Analysis | `scaffold_activity_cliff_analysis/` | Murcko scaffolds, scaffold splits, activity cliffs, SAR interpretation, and validation leakage. |

## Supplementary Cheminformatics Workflows

Additional worksheets covering topics not fully covered in the earlier packages:

- molecular similarity search,
- substructure and SMARTS search,
- diversity picking,
- drug-likeness filtering,
- SDF import/export and provenance,
- molecular visualization and quality control,
- R-group decomposition,
- matched molecular pairs,
- reaction enumeration,
- pharmacophore fingerprint search,
- integrated quality-control workflows,
- focused library design.

See:

```text
docs/teaching/supplementary_cheminformatics_workflows/
```

## Modern pedagogy booklet

A teacher-facing booklet is available in:

```text
orange_chemistry_booklet/
```

Files:

```text
orange_in_chemistry_pedagogy_booklet.docx
orange_in_chemistry_pedagogy_booklet.pdf
```

The booklet summarizes pedagogical guidelines, modern lesson patterns, classroom examples, assessment rubrics, accessibility notes, troubleshooting, and an implementation roadmap for using Orange Data Mining in chemistry education.


## Teacher-facing booklet companion

The Orange Chemistry Booklet folder includes an additional Exercise and Implementation Companion that consolidates all exercises and classroom instructions across the teaching packages.

Path: `docs/teaching/orange_chemistry_booklet/orange_chemistry_exercise_implementation_companion.pdf`

## Molecule QC Dashboard

Folder: `docs/teaching/molecule_qc_dashboard/`

Professional-core teaching package for molecule quality control before standardization, fingerprints, QSAR, ChEMBL curation, and docking workflows.

## Molecule Import Hub

Professional import workflows for CSV/TSV/SMI/SDF molecular data, including import reports, failed-record outputs and CLI batch import.

See: `docs/teaching/molecule_import_hub/`

## QSAR Studio Model Hub and Validation Dashboard

Folder: `docs/teaching/qsar_studio/`

The QSAR Studio teaching materials now include exercises for model training and validation:

- `worksheet_03_qsar_model_hub.md`
- `worksheet_04_qsar_validation_dashboard.md`

These worksheets connect `QSAR Dataset Builder`, descriptor/fingerprint widgets, `QSAR Model Hub`, and `QSAR Validation Dashboard` into a complete QSAR modeling and validation workflow.
