# Worksheet 02 — Williams Leverage

## Context

Williams leverage measures how unusual a compound is in the descriptor space used by a regression model. The common threshold is:

```text
h* = 3(p + 1) / n
```

where `p` is the number of descriptors and `n` is the number of reference compounds.

## Intended learning outcomes

Students will be able to:

1. Explain leverage in descriptor-space terms.
2. Interpret `AD_leverage` and `AD_in_williams`.
3. Relate high leverage to extrapolation risk.

## Orange workflow

```text
File(reference) ─────────────┐
                             ↓ Reference Data
File(query) → Applicability Domain → Data Results → Data Table
```

## Settings

- Williams leverage: ON
- kNN distance: OFF
- Mahalanobis distance: OFF

## Student tasks

1. Run the widget using only Williams leverage.
2. Open `AD Summary`.
3. Record the value of `h*`.
4. Open `Data Results`.
5. Sort compounds by `AD_leverage` from high to low.
6. Identify the top three highest-leverage compounds.

## Guiding questions

1. What does a high leverage value indicate?
2. Which descriptors are extreme for the high-leverage compounds?
3. Does high leverage always mean the compound is chemically impossible?
4. Why should high-leverage predictions be flagged?

## Expected output columns

- `AD_leverage`
- `AD_in_williams`
- `AD_in_domain`

## Teacher note

Williams leverage is especially useful for linear regression and MLR-style QSAR. It is less directly tied to non-linear models, but it remains a useful descriptor-space diagnostic.
