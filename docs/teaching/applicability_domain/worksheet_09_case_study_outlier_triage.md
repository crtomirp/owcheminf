# Worksheet 09 — Outlier Triage Case Study

## Context

In real screening projects, outside-domain molecules are not automatically discarded. They must be triaged: some are errors, some are novel, and some are simply outside the model scope.

## Intended learning outcomes

Students will be able to:

1. Classify outside-domain molecules into practical categories.
2. Distinguish data errors from genuine chemical novelty.
3. Recommend next steps for outlier compounds.

## Orange workflow

```text
File(query) → Applicability Domain → Data Results → Data Table
```

Optional:

```text
File(query) → Cyclic Registry Fingerprint → Matched Registry Entries
```

## Student tasks

For each outside-domain compound, assign one label:

- likely data/preprocessing issue,
- outside model scope,
- chemically novel but plausible,
- should be manually reviewed,
- should be excluded from model-based ranking.

## Guiding questions

1. Is the SMILES valid and standardized?
2. Are descriptor values chemically plausible?
3. Is the compound much larger or more polar than the training set?
4. Does it contain a new scaffold or unusual functional group?
5. Should the training set be expanded to include this chemistry?

## Deliverable

Prepare a table:

| Compound | AD status | Reason | Recommended action |
|---|---|---|---|
|  |  |  |  |

## Teacher note

This activity emphasizes that AD is a decision-support tool, not a fully automated rejection rule.
