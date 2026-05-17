# Detecting Activity Cliffs with Fingerprints

## Context

Activity cliff detection depends on molecular similarity. Different fingerprints and thresholds can change which pairs are detected.

## Intended learning outcomes

Students can:

- explain why similarity thresholds matter,
- test different threshold settings,
- discuss sensitivity of cliff detection.

## Orange workflow

```text
File → Fingerprint Generator → Activity Cliff Finder → Cliff Pairs
```

Alternative conceptual comparison:

```text
File → Cyclic Registry Fingerprint → Matched Registry Entries
```

## Tasks

1. Run Activity Cliff Finder with similarity threshold 0.60.
2. Repeat with 0.75 or 0.80.
3. Compare the number of cliff pairs.
4. Discuss whether stricter similarity creates more chemically convincing cliffs.

## Guiding questions

1. What happens when the similarity threshold is too low?
2. What happens when it is too high?
3. Can a pair be chemically similar but fingerprint-dissimilar?
4. How would you report threshold choices?

## Expected output

Students should learn that activity cliff detection is threshold-dependent and must be reported transparently.

## Assessment rubric

| Criterion | Basic | Good | Excellent |
|---|---|---|---|
| Workflow execution | Runs the main widgets | Correct settings and outputs | Clear, reproducible workflow |
| Chemical interpretation | Minimal description | Correct scaffold/cliff explanation | Insightful SAR reasoning |
| Validation discussion | Mentions train/test split | Explains random vs scaffold split | Connects validation, AD, and cliffs |
| Reporting | Fragmentary notes | Clear short answers | Publication-style transparency |
