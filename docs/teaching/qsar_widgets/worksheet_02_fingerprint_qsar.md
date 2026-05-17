# Worksheet 02: Fingerprints for QSAR

**Estimated time:** 60–90 min  
**Level:** introductory to intermediate  
**Main widgets:** Fingerprint Generator, Cyclic Registry Fingerprint, Data Table

## Context

Fingerprints encode molecular structure as binary vectors. They are widely used in similarity search and QSAR modelling.

## Intended learning outcomes

Students will be able to:

1. generate Morgan, RDKit, MACCS, and cyclic registry fingerprints,
2. compare hashed and registry-enhanced fingerprints,
3. explain why binary fingerprints are useful in machine learning,
4. inspect interpretable registry matches.

## Orange workflows

General fingerprint:

```text
File → Fingerprint Generator → Data Table
```

Registry-enhanced fingerprint:

```text
File → Cyclic Registry Fingerprint → Data Table
Cyclic Registry Fingerprint → Matched Registry Entries → Data Table
```

## CLI alternative for cyclic registry fingerprints

```bash
owcheminf-cyclic-registry-fingerprint \
  examples/qsar_widgets/qsar_training_set.csv \
  --smiles-column smiles \
  --name-column name \
  --out-prefix outputs/qsar_crfp \
  --write-full-matrix
```

## Student tasks

1. Generate a Morgan fingerprint.
2. Generate a cyclic registry fingerprint.
3. Compare the number of output columns.
4. Inspect `Matched Registry Entries` for caffeine, nicotine, and aspirin.
5. Write a short explanation of why the registry output is more interpretable.

## Guiding questions

- What is a bit collision?
- Which output is more useful for model fitting?
- Which output is more useful for chemical explanation?
- Why can two chemical motifs activate the same bit?

## Expected output

A fingerprint table and a registry-match table that links molecules to detected cyclic or functional motifs.
