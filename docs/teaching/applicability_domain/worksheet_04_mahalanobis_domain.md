# Worksheet 04 — Mahalanobis Distance Domain

## Context

Mahalanobis distance measures how far a compound is from the multivariate center of the reference descriptor distribution, accounting for descriptor covariance.

## Intended learning outcomes

Students will be able to:

1. Explain why multivariate distance is different from one-descriptor thresholds.
2. Interpret `AD_maha_d2` and `AD_in_maha`.
3. Discuss when Mahalanobis distance may fail or become unstable.

## Orange workflow

```text
File(reference) ─────────────┐
                             ↓ Reference Data
File(query) → Applicability Domain → Data Results → Data Table
```

## Settings

- Williams leverage: OFF
- kNN distance: OFF
- Mahalanobis distance: ON
- Mahalanobis α: 0.95
- Use chi-square threshold: ON

## Student tasks

1. Run the widget with Mahalanobis only.
2. Record the Mahalanobis threshold from `AD Summary`.
3. Sort query compounds by `AD_maha_d2`.
4. Identify compounds outside the Mahalanobis domain.
5. Repeat with α = 0.90.

## Guiding questions

1. Which compounds have the largest Mahalanobis distance?
2. How does α affect strictness?
3. Why can Mahalanobis distance be problematic with many descriptors and few molecules?
4. Why do correlated descriptors matter?

## Teacher note

Mahalanobis distance can be unstable when the descriptor covariance matrix is poorly conditioned. This is common in QSAR datasets with many descriptors and few training compounds.
