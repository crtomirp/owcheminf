# Worksheet 08 — Reporting Applicability Domain in QSAR

## Context

A QSAR report should include model performance and applicability domain. Without AD, external predictions may look more certain than they really are.

## Intended learning outcomes

Students will be able to:

1. Write a clear AD methods paragraph.
2. Report thresholds and number of in-domain compounds.
3. Distinguish between model validation and prediction screening.

## Orange workflow

```text
File(reference) ─────────────┐
                             ↓ Reference Data
File(query) → Applicability Domain → AD Summary → Data Table
```

## Student tasks

1. Run the AD widget.
2. Open `AD Summary`.
3. Record:
   - number of reference compounds,
   - number of query compounds,
   - number of features,
   - Williams `h*`,
   - kNN threshold,
   - number of query compounds inside domain.
4. Write a methods paragraph.
5. Write a results paragraph.

## Template methods paragraph

```text
The applicability domain was evaluated in the descriptor space used for modeling. Williams leverage and k-nearest-neighbor distance were applied to the reference training set. Query compounds were considered inside the applicability domain only when they satisfied the selected AD criteria.
```

## Template results paragraph

```text
Out of N query compounds, M were inside the applicability domain. Outside-domain compounds were flagged as extrapolations and were not used for final ranking without additional expert review.
```

## Assessment

A good answer must include the descriptor space, AD methods, thresholds, and decision rule.
