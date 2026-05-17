# Applicability Domain Workbench

**Category:** Cheminf - Modeling  
**Widget:** Applicability Domain Workbench

The Applicability Domain Workbench evaluates whether query compounds are inside the descriptor/fingerprint space represented by a reference training set. It is intended for QSAR reliability assessment and external prediction triage.

## Inputs

- **Query Data**: compounds to score.
- **Reference Data**: training/reference compounds used to define the domain. If omitted, Query Data is used as its own reference.

## Outputs

- **AD Results**: query records with AD metrics and flags.
- **Reference Results**: reference records scored against the fitted AD.
- **Out-of-Domain Records**: query records failing one or more enabled AD rules.
- **AD Summary**: compact summary table.
- **Method Details**: settings and feature preview.

## Methods

The workbench supports:

- Williams leverage
- kNN distance domain
- Mahalanobis distance domain
- AND/OR rule combination

## Recommended workflow

```text
QSAR Dataset Builder → Mol Descriptors 2 / Fingerprint Generator
                    → QSAR Model Hub
                    → Applicability Domain Workbench
```

## CLI

```bash
owcheminf-ad-workbench \
  examples/qsar_studio/qsar_ad_explanation_demo.csv \
  examples/qsar_studio/qsar_ad_query_demo.csv \
  --out-prefix outputs/ad_workbench_demo
```
