# CLI and Non-Orange Use for QSAR Teaching

A CLI is currently available for the cyclic registry fingerprint:

```bash
owcheminf-cyclic-registry-fingerprint --help
```

Example:

```bash
owcheminf-cyclic-registry-fingerprint \
  examples/qsar_widgets/qsar_training_set.csv \
  --smiles-column smiles \
  --name-column name \
  --out-prefix outputs/qsar_crfp \
  --write-full-matrix \
  --write-json
```

For full QSAR modelling, Orange is still recommended because it makes splitting, model comparison, and output inspection visible. Future CLI commands could include descriptor calculation, scaffold splitting, QSAR regression, and activity cliff export.
