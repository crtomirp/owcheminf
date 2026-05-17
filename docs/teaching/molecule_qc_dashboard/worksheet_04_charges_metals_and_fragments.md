# Worksheet 4: Charges, Metals, and Fragments

## Context

Charged molecules, salts, counterions, and metal-containing records are common in real datasets.

## Workflow

```text
File → Molecule QC Dashboard → QC Report → Data Table
```

## Tasks

1. Identify records with `NET_FORMAL_CHARGE`.
2. Identify records with `MULTI_FRAGMENT`.
3. Identify records with `METAL_PRESENT`.
4. Discuss which records can be fixed automatically.

## Questions

1. Is a formal charge always an error?
2. Why are salts common in drug datasets?
3. Why are metals problematic for many RDKit descriptors?
4. Why should docking pose data use different QC rules than QSAR data?

## Expected product

A domain-specific QC rule set for drug-like QSAR datasets.
