# QSAR Report Generator

**Category:** Cheminf - QSAR  
**Widget:** QSAR Report Generator

The QSAR Report Generator collects dataset, metrics, prediction, validation, applicability-domain and model-explanation outputs into a reproducible Markdown/HTML report.
It also renders interactive diagnostics for observed-vs-predicted values, residuals, core metrics and top descriptors.
The widget can also export the rendered report directly to PDF from the control panel.

## Inputs

- Dataset
- Metrics
- Predictions
- Validation Summary
- Feature Importance
- Model Summary
- AD Summary
- Explanation Summary

All inputs are optional, but the report becomes more useful as more QSAR Studio tables are connected.
The widget auto-detects common column aliases such as `observed`/`actual`/`y_true`,
`predicted`/`prediction`/`predicted_pActivity`, and both long or wide metric tables.

## Outputs

- Report Markdown
- Report HTML
- Report PDF Path
- Report Sections
- Report Summary

## Typical workflow

```text
QSAR/QSPR Model Hub → Predictions ─────────────→ QSAR Report Generator / Predictions
QSAR/QSPR Model Hub → Metrics ─────────────────→ QSAR Report Generator / Metrics
QSAR/QSPR Model Hub → Model Summary ───────────→ QSAR Report Generator / Model Summary
Original QSAR-ready descriptor table ──────────→ QSAR Report Generator / Dataset

Optional extended diagnostics:

QSAR Validation Dashboard → Validation Summary ─→ QSAR Report Generator / Validation Summary
Applicability Domain Workbench → Summary ──────→ QSAR Report Generator / AD Summary
Model Explanation → Summary ───────────────────→ QSAR Report Generator / Explanation Summary
Model Explanation → Feature Importance ────────→ QSAR Report Generator / Feature Importance
```

## CLI

```bash
owcheminf-qsar-report-generator \
  --dataset examples/qsar_studio/qsar_model_hub_demo.csv \
  --metrics examples/qsar_studio/qsar_report_metrics_demo.csv \
  --predictions examples/qsar_studio/qsar_predictions_demo.csv \
  --out-prefix outputs/qsar_report_demo
```
