# Non-Orange and API Notes

The current teaching package is primarily designed for Orange workflows. However, several tasks can be performed outside Orange.

## Already available CLI tools

The package includes a command-line tool for the Cyclic Registry Fingerprint:

```bash
owcheminf-cyclic-registry-fingerprint --help
```

This can be used after ChEMBL curation to generate interpretable fingerprints from a CSV file.

## Example CLI workflow after curation

```bash
owcheminf-cyclic-registry-fingerprint \
  examples/chembl_bioactivity_curation/chembl_qsar_curated_example.csv \
  --smiles-column canonical_smiles \
  --name-column molecule_id \
  --out-prefix outputs/chembl_curated_crfp \
  --write-full-matrix \
  --write-json
```

## Future CLI direction

A future CLI could automate retrieval and curation:

```bash
owcheminf-chembl-curate \
  --target CHEMBLxxxx \
  --activity-type IC50 \
  --min-confidence 8 \
  --relation '=' \
  --out curated.csv \
  --log curation_log.json
```

For teaching, it is often better to perform the curation interactively in Orange first, so that students can see the consequences of each filter.
