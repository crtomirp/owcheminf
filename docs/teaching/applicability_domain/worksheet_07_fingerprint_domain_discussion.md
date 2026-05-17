# Worksheet 07 — Applicability Domain for Fingerprints

## Context

The current Applicability Domain widget uses continuous descriptor columns. Fingerprints are often binary and high-dimensional, so AD interpretation requires additional care.

## Intended learning outcomes

Students will be able to:

1. Discuss the difference between descriptor-space AD and fingerprint-space similarity.
2. Explain why Tanimoto similarity is often used with fingerprints.
3. Connect cyclic registry fingerprint matches with qualitative domain interpretation.

## Orange workflow

```text
File → Cyclic Registry Fingerprint → Matched Registry Entries → Data Table
```

For continuous AD:

```text
File → Mol Descriptors 2 → Applicability Domain
```

## Student tasks

1. Generate cyclic registry fingerprint output.
2. Examine `Matched Registry Entries` for reference and query compounds.
3. Compare whether outside-domain descriptor outliers also have unusual registry motifs.
4. Discuss whether the registry matches support or weaken confidence in prediction.

## Guiding questions

1. If a query compound has no registry motifs seen in the training set, should we trust the model?
2. Is this the same as being outside Williams leverage AD?
3. Why is Tanimoto similarity often preferred for binary fingerprints?
4. How could a future AD widget support fingerprint-based AD?

## Teacher note

This worksheet is conceptual. It helps students understand that AD depends on representation: descriptors, fingerprints, scaffolds, or learned embeddings can all define different notions of domain.
