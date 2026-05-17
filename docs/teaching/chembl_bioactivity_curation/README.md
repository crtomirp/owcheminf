# Teaching Package B: ChEMBL Bioactivity Curation

This teaching package supports practical lessons on retrieving, cleaning, filtering, and documenting bioactivity data from ChEMBL for QSAR/QSPR modelling.

The package is designed for the following widgets:

- **ChEMBL Browser**
- **ChEMBL Bioactivity Retriever**
- **Mol Standardizer**
- **Mol Descriptors 2**
- **Fingerprint Generator**
- **Cyclic Registry Fingerprint**
- **Scaffold Splitter** / **Scaffold Analysis**
- **QSAR Regression**
- **Applicability Domain**

## Main learning idea

A QSAR model is only as reliable as its curated input data. Students learn that downloading bioactivity data is not the same as preparing a QSAR-ready dataset. They must inspect targets, assay types, units, relations, duplicated measurements, molecular structures, and data provenance.

## Recommended sequence

1. Worksheet 01 — What ChEMBL data represents
2. Worksheet 02 — Target search and target confidence
3. Worksheet 03 — Bioactivity endpoints, units, and relations
4. Worksheet 04 — Retrieving bioactivity data
5. Worksheet 05 — Assay and confidence filtering
6. Worksheet 06 — Molecular structure cleaning after retrieval
7. Worksheet 07 — Duplicate handling and value aggregation
8. Worksheet 08 — Building a QSAR-ready pChEMBL dataset
9. Worksheet 09 — Provenance and FAIR reporting
10. Worksheet 10 — Capstone ChEMBL curation project

## Example data

The folder `examples/chembl_bioactivity_curation/` contains small teaching datasets. Some files are synthetic or simplified so that students can practice curation logic without relying on live network access.

## Suggested Orange workflow

```text
ChEMBL Browser
  → ChEMBL Bioactivity Retriever
  → Mol Standardizer
  → Mol Descriptors 2 / Fingerprint Generator / Cyclic Registry Fingerprint
  → Scaffold Splitter
  → QSAR Regression
  → Applicability Domain
```

## Important teaching note

Live ChEMBL results may change over time. Students should record the query date, target identifier, filters, assay selection criteria, and any manual curation decisions.
