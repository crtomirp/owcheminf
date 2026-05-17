# Worksheet 03 — kNN Distance Applicability Domain

## Context

The k-nearest-neighbor AD asks whether a query molecule has close neighbors in the reference descriptor space. A compound far away from all training compounds is likely an extrapolation.

## Intended learning outcomes

Students will be able to:

1. Explain kNN distance as a local similarity criterion.
2. Interpret `AD_knn_dist` and `AD_in_knn`.
3. Explore how `k` and the quantile threshold affect AD classification.

## Orange workflow

```text
File(reference) ─────────────┐
                             ↓ Reference Data
File(query) → Applicability Domain → Data Results → Data Table
```

## Settings

Start with:

- Williams leverage: OFF
- kNN distance: ON
- kNN k: 5
- kNN quantile: 0.95
- Mahalanobis distance: OFF

Then repeat with:

- kNN k: 3
- kNN quantile: 0.90

## Student tasks

1. Run the widget with kNN only.
2. Record the kNN threshold from `AD Summary`.
3. Sort query compounds by `AD_knn_dist`.
4. Repeat with stricter settings.
5. Compare which compounds change status.

## Guiding questions

1. Which compounds are farthest from the reference set?
2. How does lowering the quantile threshold affect the number of outside-domain compounds?
3. Why might kNN be useful for non-linear QSAR models?
4. Is kNN more local or global than Williams leverage?

## Expected observation

Changing the kNN quantile from 0.95 to 0.90 should make the domain stricter.

## Extension

Plot `MW` vs `LogP` and color by `AD_in_knn` to visualize local outliers.
