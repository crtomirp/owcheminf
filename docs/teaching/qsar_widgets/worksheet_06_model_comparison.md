# Worksheet 06: Comparing Descriptor Families and Models

**Estimated time:** 90–120 min  
**Level:** intermediate  
**Main widgets:** Fingerprint Generator, Cyclic Registry Fingerprint, Mol Descriptors 2, QSAR Regression

## Context

Different molecular representations can lead to different model performance and interpretability.

## Intended learning outcomes

Students will be able to:

1. compare descriptor-based and fingerprint-based QSAR workflows,
2. distinguish predictive performance from interpretability,
3. identify when a simpler model is preferable,
4. summarize model comparison results.

## Orange workflows

Descriptor model:

```text
File → Mol Descriptors 2 → QSAR Regression
```

Morgan fingerprint model:

```text
File → Fingerprint Generator → QSAR Regression
```

Registry fingerprint model:

```text
File → Cyclic Registry Fingerprint → QSAR Regression
```

## Student tasks

1. Build at least two models using different representations.
2. Record available metrics.
3. Compare interpretability.
4. Inspect matched registry entries for selected compounds.
5. Decide which model you would present in a report and why.

## Guiding questions

- Which representation gave the best numerical performance?
- Which representation was easiest to explain chemically?
- Would you choose the highest-scoring model automatically?
- What extra validation would you need for publication?

## Expected output

A comparison table prepared by the student with representation, model type, metrics, and interpretation notes.
