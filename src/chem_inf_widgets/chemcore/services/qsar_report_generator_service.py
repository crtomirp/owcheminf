from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class QSARReportConfig:
    title: str = "QSAR Studio Report"
    project_name: str = "QSAR project"
    author: str = ""
    include_dataset_preview: bool = True
    include_predictions_preview: bool = True
    max_preview_rows: int = 12
    include_limitations: bool = True


@dataclass(frozen=True)
class QSARReportResult:
    markdown: str
    html: str
    sections: pd.DataFrame
    summary: dict[str, Any]


def _safe_shape(df: Optional[pd.DataFrame]) -> tuple[int, int]:
    if df is None:
        return (0, 0)
    return (int(len(df)), int(len(df.columns)))


def _metric_rows(metrics: Optional[pd.DataFrame]) -> list[dict[str, Any]]:
    if metrics is None or metrics.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        d = {str(k): row[k] for k in metrics.columns}
        rows.append(d)
    return rows


def _df_preview_markdown(df: Optional[pd.DataFrame], max_rows: int) -> str:
    if df is None or df.empty:
        return "_No table was provided._"
    preview = df.head(max_rows).copy()
    try:
        return preview.to_markdown(index=False)
    except Exception:
        return "```\n" + preview.to_csv(index=False) + "```"


def _summarize_numeric(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    desc = numeric.describe().T.reset_index().rename(columns={"index": "feature"})
    keep = [c for c in ["feature", "count", "mean", "std", "min", "50%", "max"] if c in desc.columns]
    return desc[keep]


def _table_to_html(df: Optional[pd.DataFrame], max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p><em>No table was provided.</em></p>"
    return df.head(max_rows).to_html(index=False, escape=True, border=0, classes="report-table")


def generate_qsar_report(
    *,
    dataset: Optional[pd.DataFrame] = None,
    metrics: Optional[pd.DataFrame] = None,
    predictions: Optional[pd.DataFrame] = None,
    validation_summary: Optional[pd.DataFrame] = None,
    ad_summary: Optional[pd.DataFrame] = None,
    explanation_summary: Optional[pd.DataFrame] = None,
    config: QSARReportConfig | None = None,
) -> QSARReportResult:
    config = config or QSARReportConfig()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dataset_shape = _safe_shape(dataset)
    predictions_shape = _safe_shape(predictions)
    metrics_shape = _safe_shape(metrics)

    sections: list[dict[str, Any]] = []
    md: list[str] = []
    md.append(f"# {config.title}")
    md.append("")
    md.append(f"**Project:** {config.project_name}")
    if config.author:
        md.append(f"**Author:** {config.author}")
    md.append(f"**Generated:** {created}")
    md.append("")

    md.append("## 1. Executive summary")
    md.append("")
    md.append(f"- Dataset rows/columns: **{dataset_shape[0]} / {dataset_shape[1]}**")
    md.append(f"- Prediction rows/columns: **{predictions_shape[0]} / {predictions_shape[1]}**")
    md.append(f"- Metric rows/columns: **{metrics_shape[0]} / {metrics_shape[1]}**")
    md.append("- Recommended interpretation: use model metrics together with applicability-domain and explanation outputs.")
    sections.append({"section": "Executive summary", "status": "created", "rows": 1, "notes": "High-level project summary."})

    md.append("")
    md.append("## 2. Dataset overview")
    md.append("")
    if dataset is None or dataset.empty:
        md.append("No dataset table was supplied to the report generator.")
        sections.append({"section": "Dataset overview", "status": "missing", "rows": 0, "notes": "No dataset table supplied."})
    else:
        md.append(f"The dataset contains **{len(dataset)}** records and **{len(dataset.columns)}** columns.")
        numeric_summary = _summarize_numeric(dataset)
        if not numeric_summary.empty:
            md.append("")
            md.append("### Numeric descriptor/endpoint summary")
            md.append(_df_preview_markdown(numeric_summary, config.max_preview_rows))
        if config.include_dataset_preview:
            md.append("")
            md.append("### Dataset preview")
            md.append(_df_preview_markdown(dataset, config.max_preview_rows))
        sections.append({"section": "Dataset overview", "status": "created", "rows": len(dataset), "notes": "Dataset summary and preview."})

    md.append("")
    md.append("## 3. Model metrics")
    md.append("")
    if metrics is None or metrics.empty:
        md.append("No metric table was supplied.")
        sections.append({"section": "Model metrics", "status": "missing", "rows": 0, "notes": "No model metrics supplied."})
    else:
        md.append(_df_preview_markdown(metrics, config.max_preview_rows))
        sections.append({"section": "Model metrics", "status": "created", "rows": len(metrics), "notes": "Model and validation metrics."})

    md.append("")
    md.append("## 4. Prediction diagnostics")
    md.append("")
    if predictions is None or predictions.empty:
        md.append("No prediction table was supplied.")
        sections.append({"section": "Prediction diagnostics", "status": "missing", "rows": 0, "notes": "No prediction table supplied."})
    else:
        pred_cols = set(map(str, predictions.columns))
        useful = [c for c in ["observed", "predicted", "residual", "split", "compound_id", "id"] if c in pred_cols]
        md.append(f"Prediction table contains **{len(predictions)}** records.")
        if useful:
            md.append(f"Key diagnostic columns detected: {', '.join(useful)}.")
        if config.include_predictions_preview:
            md.append("")
            md.append(_df_preview_markdown(predictions, config.max_preview_rows))
        sections.append({"section": "Prediction diagnostics", "status": "created", "rows": len(predictions), "notes": "Prediction table preview."})

    optional_sections = [
        ("5. Validation dashboard", validation_summary, "Validation summary from QSAR Validation Dashboard."),
        ("6. Applicability domain", ad_summary, "Applicability-domain summary."),
        ("7. Model explanation", explanation_summary, "Feature/explanation summary."),
    ]
    for title, table, notes in optional_sections:
        md.append("")
        md.append(f"## {title}")
        md.append("")
        if table is None or table.empty:
            md.append("No table was supplied for this section.")
            sections.append({"section": title, "status": "missing", "rows": 0, "notes": notes})
        else:
            md.append(_df_preview_markdown(table, config.max_preview_rows))
            sections.append({"section": title, "status": "created", "rows": len(table), "notes": notes})

    md.append("")
    md.append("## 8. Reproducibility checklist")
    md.append("")
    checklist = [
        "Input dataset source and version recorded.",
        "Molecular standardization protocol recorded.",
        "Descriptor/fingerprint settings recorded.",
        "Train/test or cross-validation strategy recorded.",
        "Applicability-domain method and thresholds recorded.",
        "Software/package versions recorded.",
    ]
    for item in checklist:
        md.append(f"- [ ] {item}")
    sections.append({"section": "Reproducibility checklist", "status": "created", "rows": len(checklist), "notes": "Manual checklist for reporting."})

    if config.include_limitations:
        md.append("")
        md.append("## 9. Limitations and responsible use")
        md.append("")
        md.append("QSAR predictions should not be interpreted as experimental facts. They are model-based estimates and depend on dataset quality, molecular standardization, descriptor choice, validation strategy, and applicability domain. Out-of-domain predictions should be flagged and treated as lower-confidence hypotheses.")
        sections.append({"section": "Limitations", "status": "created", "rows": 1, "notes": "Responsible-use statement."})

    markdown = "\n".join(md) + "\n"
    html_body = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html_body = html_body.replace("\n", "<br/>\n")
    html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>{config.title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; line-height: 1.45; color: #172033; }}
h1 {{ color: #0f3b5f; border-bottom: 4px solid #14b8a6; padding-bottom: .4rem; }}
h2 {{ color: #0f3b5f; margin-top: 1.8rem; }}
code, pre {{ background: #f3f4f6; padding: .1rem .25rem; border-radius: .25rem; }}
.report-table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
.report-table th, .report-table td {{ border: 1px solid #d1d5db; padding: .35rem .5rem; }}
</style></head><body>{html_body}</body></html>"""

    summary = {
        "title": config.title,
        "project_name": config.project_name,
        "created_utc": created,
        "dataset_rows": dataset_shape[0],
        "dataset_columns": dataset_shape[1],
        "prediction_rows": predictions_shape[0],
        "metric_rows": metrics_shape[0],
        "sections_created": int(sum(1 for s in sections if s["status"] == "created")),
        "sections_missing": int(sum(1 for s in sections if s["status"] == "missing")),
    }
    return QSARReportResult(markdown=markdown, html=html, sections=pd.DataFrame(sections), summary=summary)


def write_report_files(result: QSARReportResult, out_prefix: str | Path) -> dict[str, str]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": str(prefix.with_suffix(".report.md")),
        "html": str(prefix.with_suffix(".report.html")),
        "sections": str(prefix.with_suffix(".sections.csv")),
        "summary": str(prefix.with_suffix(".summary.json")),
    }
    Path(paths["markdown"]).write_text(result.markdown, encoding="utf-8")
    Path(paths["html"]).write_text(result.html, encoding="utf-8")
    result.sections.to_csv(paths["sections"], index=False)
    Path(paths["summary"]).write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    return paths
