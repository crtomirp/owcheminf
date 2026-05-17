# QSAR Report Generator

**Category:** Cheminf - Modeling  
**Widget:** QSAR Report Generator

The QSAR Report Generator collects dataset, metrics, prediction, validation, applicability-domain and model-explanation outputs into a reproducible Markdown/HTML report.

## Inputs

- Dataset
- Metrics
- Predictions
- Validation Summary
- AD Summary
- Explanation Summary

All inputs are optional, but the report becomes more useful as more QSAR Studio tables are connected.

## Outputs

- Report Markdown
- Report HTML
- Report Sections
- Report Summary

## Typical workflow

```text
QSAR Model Hub → Predictions ─────────────┐
QSAR Model Hub → Metrics ─────────────────┤
QSAR Validation Dashboard → Summary ──────┤
Applicability Domain Workbench → Summary ─┤→ QSAR Report Generator
Model Explanation → Summary ──────────────┘
```

## CLI

```bash
owcheminf-qsar-report-generator \
  --dataset examples/qsar_studio/qsar_model_hub_demo.csv \
  --metrics examples/qsar_studio/qsar_report_metrics_demo.csv \
  --predictions examples/qsar_studio/qsar_predictions_demo.csv \
  --out-prefix outputs/qsar_report_demo
```
