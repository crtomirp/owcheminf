# CLI usage: Cyclic Registry Fingerprint without Orange

The package provides a command-line interface for the same 4096-bit fingerprint used by the Orange widget.

## Basic CSV example

```bash
owcheminf-cyclic-registry-fingerprint \
  examples/cyclic_registry_fingerprint/cyclic_registry_training_set.csv \
  --smiles-column smiles \
  --name-column name \
  --out-prefix outputs/training_set_crfp \
  --write-json
```

## SMI/TXT example

Input file format:

```text
c1ccncc1 pyridine
c1ncc[nH]1 imidazole
Cn1cnc2c1c(=O)n(C)c(=O)n2C caffeine
```

Run:

```bash
owcheminf-cyclic-registry-fingerprint \
  examples/cyclic_registry_fingerprint/mini_heterocycles.smi \
  --out-prefix outputs/mini_heterocycles_crfp \
  --write-json
```

## SDF example

```bash
owcheminf-cyclic-registry-fingerprint molecules.sdf \
  --format sdf \
  --out-prefix outputs/molecules_crfp \
  --write-json
```

## Writing the full 4096-bit matrix

By default, the CLI writes a compact active-bit table. To write a wide 4096-bit table:

```bash
owcheminf-cyclic-registry-fingerprint input.csv \
  --smiles-column smiles \
  --out-prefix outputs/full_matrix_example \
  --write-full-matrix
```

## Important options

| Option | Meaning |
|---|---|
| `--smiles-column` | Name of the SMILES column in CSV/TSV files. |
| `--id-column` | Optional ID column. |
| `--name-column` | Optional compound name column. |
| `--no-morgan` | Disable the Morgan section and keep only registry/topology sections. |
| `--no-sanitize` | Parse input without standard RDKit sanitization where possible. |
| `--no-atom-matches` | Do not write atom index tuples in the matches file. |
| `--write-full-matrix` | Write all 4096 bit columns. |
| `--write-json` | Write provenance and summary JSON. |
| `--fail-on-invalid` | Return exit code 2 if any molecule fails parsing. |

## Relationship to the Orange widget

The CLI and the Orange widget use the same service layer:

```text
chem_inf_widgets.chemcore.descriptors.cyclic_registry_fingerprint
```

Therefore, they should produce the same active bits and registry matches for the same molecules and settings.
