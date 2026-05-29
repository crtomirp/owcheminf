from __future__ import annotations

import html as _html
import json
import math
from dataclasses import dataclass
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
    include_auto_conclusions: bool = True
    include_oecd_readiness: bool = True
    include_publication_checklist: bool = True


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


def _norm_key(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _metric_lookup(metrics: Optional[pd.DataFrame]) -> dict[tuple[str, str], float]:
    """Return {(split, metric): value}; split may be '' when not supplied."""
    out: dict[tuple[str, str], float] = {}
    if metrics is None or metrics.empty:
        return out
    cols = {_norm_key(c): c for c in metrics.columns}
    metric_col = cols.get("metric") or cols.get("name")
    value_col = cols.get("value") or cols.get("score")
    split_col = cols.get("split") or cols.get("dataset") or cols.get("partition")
    if metric_col and value_col:
        for _, row in metrics.iterrows():
            try:
                value = float(row[value_col])
            except Exception:
                continue
            split = _norm_key(row[split_col]) if split_col else ""
            metric = _norm_key(row[metric_col])
            out[(split, metric)] = value
            out.setdefault(("", metric), value)
        return out
    # Wide table fallback: r2_test, rmse_cv, train_r2, etc.
    for col in metrics.columns:
        try:
            value = float(metrics[col].iloc[0])
        except Exception:
            continue
        key = _norm_key(col)
        split = ""
        metric = key
        for s in ("train", "test", "cv", "external", "validation"):
            if key.startswith(s + "_"):
                split, metric = s, key[len(s) + 1:]
                break
            if key.endswith("_" + s):
                split, metric = s, key[:-(len(s) + 1)]
                break
        out[(split, metric)] = value
        out.setdefault(("", metric), value)
    return out


def _first_metric(lookup: dict[tuple[str, str], float], names: list[str], splits: list[str]) -> float | None:
    aliases: list[str] = []
    for n in names:
        k = _norm_key(n)
        aliases.extend([k, k.replace("2", "_2"), k.replace("_", "")])
    for split in splits:
        sk = _norm_key(split)
        for alias in aliases:
            if (sk, alias) in lookup:
                return lookup[(sk, alias)]
    for alias in aliases:
        if ("", alias) in lookup:
            return lookup[("", alias)]
    return None


def _fmt_num(v: Any, digits: int = 3) -> str:
    try:
        f = float(v)
    except Exception:
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.{digits}f}"


def _classify_r2(r2: float | None) -> str:
    if r2 is None:
        return "not available"
    if r2 >= 0.80:
        return "strong"
    if r2 >= 0.60:
        return "moderate"
    if r2 >= 0.40:
        return "weak-to-moderate"
    return "weak"


def _detect_column(df: Optional[pd.DataFrame], candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    low = {_norm_key(c): c for c in df.columns}
    for c in candidates:
        if _norm_key(c) in low:
            return low[_norm_key(c)]
    return None


def _prediction_diagnostics(predictions: Optional[pd.DataFrame]) -> dict[str, Any]:
    if predictions is None or predictions.empty:
        return {}
    obs = _detect_column(predictions, ["observed", "y_true", "actual", "experimental", "pActivity"])
    pred = _detect_column(predictions, ["predicted", "y_pred", "prediction", "predicted_pActivity"])
    split = _detect_column(predictions, ["split", "dataset", "partition"])
    residual = _detect_column(predictions, ["residual", "error", "prediction_error"])
    out: dict[str, Any] = {"observed_column": obs, "predicted_column": pred, "split_column": split, "residual_column": residual}
    if obs and pred:
        tmp = predictions[[obs, pred]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(tmp) >= 2:
            err = tmp[pred] - tmp[obs]
            ss_res = float((err ** 2).sum())
            ss_tot = float(((tmp[obs] - tmp[obs].mean()) ** 2).sum())
            out["r2_from_predictions"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
            out["rmse_from_predictions"] = float((err ** 2).mean() ** 0.5)
            out["mae_from_predictions"] = float(err.abs().mean())
    if residual:
        res = pd.to_numeric(predictions[residual], errors="coerce").dropna()
        if len(res):
            out["residual_mean"] = float(res.mean())
            out["residual_std"] = float(res.std()) if len(res) > 1 else 0.0
            out["residual_abs_max"] = float(res.abs().max())
    if split:
        out["split_counts"] = predictions[split].astype(str).value_counts().to_dict()
    return out


def _dataset_profile(dataset: Optional[pd.DataFrame]) -> dict[str, Any]:
    if dataset is None or dataset.empty:
        return {}
    numeric = dataset.select_dtypes(include="number")
    non_numeric = [c for c in dataset.columns if c not in numeric.columns]
    smiles_col = _detect_column(dataset, ["smiles", "canonical_smiles", "SMILES"])
    id_col = _detect_column(dataset, ["compound_id", "id", "name", "molecule_id"])
    target_col = _detect_column(dataset, ["pActivity", "pIC50", "activity", "target", "y"])
    missing_total = int(dataset.isna().sum().sum())
    duplicate_ids = 0
    if id_col:
        duplicate_ids = int(dataset[id_col].astype(str).duplicated().sum())
    return {
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "numeric_columns": int(len(numeric.columns)),
        "non_numeric_columns": int(len(non_numeric)),
        "smiles_column": smiles_col,
        "id_column": id_col,
        "target_column": target_col,
        "missing_values": missing_total,
        "duplicate_ids": duplicate_ids,
    }


def _ad_profile(ad_summary: Optional[pd.DataFrame]) -> dict[str, Any]:
    if ad_summary is None or ad_summary.empty:
        return {}
    out: dict[str, Any] = {"rows": int(len(ad_summary))}
    # Summary-table style: metric/value, property/value, or wide table.
    lookup = _metric_lookup(ad_summary)
    for key in ["ad_coverage", "coverage", "in_domain_fraction", "in_domain_rate"]:
        val = _first_metric(lookup, [key], [""])
        if val is not None:
            out["coverage"] = val
            break
    bool_col = _detect_column(ad_summary, ["AD_in_domain", "in_domain", "within_ad"])
    if bool_col:
        s = ad_summary[bool_col]
        if s.dtype == bool:
            out["coverage"] = float(s.mean())
            out["out_of_domain_count"] = int((~s).sum())
        else:
            low = s.astype(str).str.lower()
            in_domain = low.isin(["true", "1", "yes", "in", "in_domain"])
            if in_domain.any():
                out["coverage"] = float(in_domain.mean())
                out["out_of_domain_count"] = int((~in_domain).sum())
    return out


def _explanation_profile(explanation_summary: Optional[pd.DataFrame]) -> dict[str, Any]:
    if explanation_summary is None or explanation_summary.empty:
        return {}
    feature_col = _detect_column(explanation_summary, ["feature", "descriptor", "name"])
    value_col = _detect_column(explanation_summary, ["importance", "value", "coefficient", "mean_abs_shap"])
    out: dict[str, Any] = {"rows": int(len(explanation_summary))}
    if feature_col:
        out["top_features"] = [str(x) for x in explanation_summary[feature_col].head(8).tolist()]
    if feature_col and value_col:
        tmp = explanation_summary[[feature_col, value_col]].copy()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.dropna().sort_values(value_col, key=lambda s: s.abs(), ascending=False).head(8)
        out["top_feature_pairs"] = [(str(r[feature_col]), float(r[value_col])) for _, r in tmp.iterrows()]
    return out


def _make_executive_bullets(dataset_profile: dict[str, Any], metrics_lookup: dict[tuple[str, str], float], pred_diag: dict[str, Any], ad_prof: dict[str, Any], expl_prof: dict[str, Any]) -> list[str]:
    test_r2 = _first_metric(metrics_lookup, ["r2", "r_2", "q2", "q_2"], ["test", "external", "validation", ""])
    test_rmse = _first_metric(metrics_lookup, ["rmse"], ["test", "external", "validation", ""])
    cv_r2 = _first_metric(metrics_lookup, ["cv_r2", "q2", "q_2", "r2"], ["cv", "cross_validation", ""])
    if test_r2 is None and pred_diag.get("r2_from_predictions") is not None:
        test_r2 = float(pred_diag["r2_from_predictions"])
    if test_rmse is None and pred_diag.get("rmse_from_predictions") is not None:
        test_rmse = float(pred_diag["rmse_from_predictions"])

    bullets = []
    if dataset_profile:
        target = dataset_profile.get("target_column") or "dependent variable"
        bullets.append(
            f"The report summarizes a QSAR/QSPR dataset with **{dataset_profile['rows']} compounds** and **{dataset_profile['numeric_columns']} numeric modeling columns**; detected target: **{target}**."
        )
    else:
        bullets.append("No source dataset was supplied, so dataset curation and descriptor coverage could not be assessed.")

    if test_r2 is not None or test_rmse is not None or cv_r2 is not None:
        parts = []
        if test_r2 is not None:
            parts.append(f"test/external R² ≈ **{_fmt_num(test_r2)}** ({_classify_r2(test_r2)} predictive signal)")
        if test_rmse is not None:
            parts.append(f"RMSE ≈ **{_fmt_num(test_rmse)}**")
        if cv_r2 is not None:
            parts.append(f"CV Q²/R² ≈ **{_fmt_num(cv_r2)}**")
        bullets.append("Main validation signal: " + "; ".join(parts) + ".")
    else:
        bullets.append("Model quality metrics were not detected; connect Metrics or Validation Summary for a publication-ready assessment.")

    if ad_prof.get("coverage") is not None:
        cov = float(ad_prof["coverage"])
        cov_pct = cov * 100 if cov <= 1.0 else cov
        bullets.append(f"Applicability-domain coverage is approximately **{cov_pct:.1f}%**; out-of-domain predictions should be highlighted in downstream use.")
    else:
        bullets.append("Applicability-domain coverage was not available; this is a key missing item for robust QSAR reporting.")

    if expl_prof.get("top_feature_pairs"):
        tops = ", ".join(name for name, _ in expl_prof["top_feature_pairs"][:5])
        bullets.append(f"Top explanatory descriptors/features detected: **{tops}**.")
    elif expl_prof.get("top_features"):
        bullets.append("Feature explanation information was supplied, but numerical importance values were not clearly detected.")
    else:
        bullets.append("No model explanation table was supplied; add feature importance, SHAP, coefficients, or fragment contributions for interpretability.")
    return bullets


def _quality_flags(dataset_profile: dict[str, Any], pred_diag: dict[str, Any], ad_prof: dict[str, Any], expl_prof: dict[str, Any]) -> list[tuple[str, str, str]]:
    flags: list[tuple[str, str, str]] = []
    if not dataset_profile:
        flags.append(("Dataset", "missing", "Connect the curated QSAR dataset."))
    else:
        flags.append(("Dataset", "ok", f"{dataset_profile['rows']} rows, {dataset_profile['numeric_columns']} numeric columns."))
        if dataset_profile.get("missing_values", 0) > 0:
            flags.append(("Missing values", "warning", f"{dataset_profile['missing_values']} missing table values detected."))
        if dataset_profile.get("duplicate_ids", 0) > 0:
            flags.append(("Duplicate identifiers", "warning", f"{dataset_profile['duplicate_ids']} duplicated IDs detected."))
    if pred_diag.get("observed_column") and pred_diag.get("predicted_column"):
        flags.append(("Prediction diagnostics", "ok", "Observed and predicted columns detected."))
    else:
        flags.append(("Prediction diagnostics", "warning", "Observed/predicted columns were not both detected."))
    if ad_prof.get("coverage") is not None:
        flags.append(("Applicability domain", "ok", "AD coverage or in-domain flags detected."))
    else:
        flags.append(("Applicability domain", "missing", "Connect AD Summary or AD Workbench output."))
    if expl_prof:
        flags.append(("Interpretability", "ok", "Explanation/feature table detected."))
    else:
        flags.append(("Interpretability", "missing", "Connect feature importance, coefficients, SHAP, or explanation summary."))
    return flags


def _flags_to_df(flags: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(flags, columns=["item", "status", "recommendation"])


def _oecd_df(dataset_profile: dict[str, Any], metrics_lookup: dict[tuple[str, str], float], ad_prof: dict[str, Any], expl_prof: dict[str, Any]) -> pd.DataFrame:
    has_metrics = bool(metrics_lookup)
    rows = [
        ("Defined endpoint", "ok" if dataset_profile.get("target_column") else "needs attention", dataset_profile.get("target_column") or "Target/dependent-variable column not detected."),
        ("Unambiguous algorithm", "needs metadata", "Report receives metrics/predictions, but algorithm metadata should be supplied from Model Hub."),
        ("Defined applicability domain", "ok" if ad_prof.get("coverage") is not None else "missing", "AD coverage/in-domain flags detected." if ad_prof.get("coverage") is not None else "Connect an AD summary."),
        ("Appropriate goodness-of-fit and predictivity", "ok" if has_metrics else "missing", "Metrics table detected." if has_metrics else "Connect train/test/CV metrics."),
        ("Mechanistic interpretation", "partial" if expl_prof else "missing", "Feature explanation detected." if expl_prof else "Connect descriptor importance/SHAP/fragment contribution table."),
    ]
    return pd.DataFrame(rows, columns=["OECD principle", "status", "evidence"])


def _simple_markdown_to_html(markdown: str) -> str:
    """Small dependency-free renderer good enough for headings, lists, fenced blocks and pipe tables."""
    lines = markdown.splitlines()
    out: list[str] = []
    in_ul = False
    in_pre = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_pre:
                out.append("<pre><code>")
                in_pre = True
            else:
                out.append("</code></pre>")
                in_pre = False
            i += 1
            continue
        if in_pre:
            out.append(_html.escape(line))
            i += 1
            continue
        if not stripped:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            i += 1
            continue
        if stripped.startswith("#"):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            level = min(len(stripped) - len(stripped.lstrip("#")), 4)
            text = stripped[level:].strip()
            out.append(f"<h{level}>{_inline_md(text)}</h{level}>")
        elif stripped.startswith("- ") or stripped.startswith("- ["):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            text = stripped[2:].strip()
            out.append(f"<li>{_inline_md(text)}</li>")
        elif stripped.startswith("|") and "|" in stripped[1:]:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            out.append(_pipe_table_to_html(table_lines))
            continue
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<p>{_inline_md(stripped)}</p>")
        i += 1
    if in_ul:
        out.append("</ul>")
    if in_pre:
        out.append("</code></pre>")
    return "\n".join(out)


def _inline_md(text: str) -> str:
    text = _html.escape(text)
    # very small bold renderer
    parts = text.split("**")
    if len(parts) > 1:
        rebuilt = []
        for idx, part in enumerate(parts):
            rebuilt.append(f"<strong>{part}</strong>" if idx % 2 else part)
        text = "".join(rebuilt)
    return text


def _pipe_table_to_html(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    h = ["<table class=\"report-table\"><thead><tr>"]
    h.extend(f"<th>{_inline_md(c)}</th>" for c in head)
    h.append("</tr></thead><tbody>")
    for row in body:
        h.append("<tr>")
        h.extend(f"<td>{_inline_md(c)}</td>" for c in row)
        h.append("</tr>")
    h.append("</tbody></table>")
    return "".join(h)


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
    metrics_lookup = _metric_lookup(metrics)
    pred_diag = _prediction_diagnostics(predictions)
    ds_prof = _dataset_profile(dataset)
    ad_prof = _ad_profile(ad_summary)
    expl_prof = _explanation_profile(explanation_summary)

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
    for bullet in _make_executive_bullets(ds_prof, metrics_lookup, pred_diag, ad_prof, expl_prof):
        md.append(f"- {bullet}")
    sections.append({"section": "Executive summary", "status": "created", "rows": 1, "notes": "Publication-style high-level summary."})

    md.append("")
    md.append("## 2. Dataset and curation overview")
    md.append("")
    if dataset is None or dataset.empty:
        md.append("No dataset table was supplied to the report generator.")
        sections.append({"section": "Dataset and curation overview", "status": "missing", "rows": 0, "notes": "No dataset table supplied."})
    else:
        md.append(f"The dataset contains **{len(dataset)}** records and **{len(dataset.columns)}** columns.")
        md.append(f"Detected identifiers/chemistry: ID column = **{ds_prof.get('id_column') or 'not detected'}**, SMILES column = **{ds_prof.get('smiles_column') or 'not detected'}**, target column = **{ds_prof.get('target_column') or 'not detected'}**.")
        md.append(f"Missing values detected in the supplied table: **{ds_prof.get('missing_values', 0)}**. Duplicate identifiers detected: **{ds_prof.get('duplicate_ids', 0)}**.")
        numeric_summary = _summarize_numeric(dataset)
        if not numeric_summary.empty:
            md.append("")
            md.append("### Numeric descriptor/endpoint summary")
            md.append(_df_preview_markdown(numeric_summary, config.max_preview_rows))
        if config.include_dataset_preview:
            md.append("")
            md.append("### Dataset preview")
            md.append(_df_preview_markdown(dataset, config.max_preview_rows))
        sections.append({"section": "Dataset and curation overview", "status": "created", "rows": len(dataset), "notes": "Dataset summary, detected columns, missing values and preview."})

    md.append("")
    md.append("## 3. Model performance and validation")
    md.append("")
    if metrics is None or metrics.empty:
        md.append("No metric table was supplied.")
        sections.append({"section": "Model performance and validation", "status": "missing", "rows": 0, "notes": "No model metrics supplied."})
    else:
        test_r2 = _first_metric(metrics_lookup, ["r2", "q2"], ["test", "external", "validation", ""])
        rmse = _first_metric(metrics_lookup, ["rmse"], ["test", "external", "validation", ""])
        md.append(f"Detected performance class: **{_classify_r2(test_r2)}** based on the available R²/Q²-like metric.")
        if test_r2 is not None or rmse is not None:
            md.append(f"Key metric summary: R²/Q²-like = **{_fmt_num(test_r2)}**, RMSE = **{_fmt_num(rmse)}**.")
        md.append("")
        md.append(_df_preview_markdown(metrics, config.max_preview_rows))
        sections.append({"section": "Model performance and validation", "status": "created", "rows": len(metrics), "notes": "Model and validation metrics with interpretation."})

    md.append("")
    md.append("## 4. Prediction diagnostics")
    md.append("")
    if predictions is None or predictions.empty:
        md.append("No prediction table was supplied.")
        sections.append({"section": "Prediction diagnostics", "status": "missing", "rows": 0, "notes": "No prediction table supplied."})
    else:
        md.append(f"Prediction table contains **{len(predictions)}** records.")
        md.append(f"Detected diagnostic columns: observed = **{pred_diag.get('observed_column') or 'not detected'}**, predicted = **{pred_diag.get('predicted_column') or 'not detected'}**, residual = **{pred_diag.get('residual_column') or 'not detected'}**, split = **{pred_diag.get('split_column') or 'not detected'}**.")
        if pred_diag.get("r2_from_predictions") is not None:
            md.append(f"Metrics recomputed from prediction rows: R² = **{_fmt_num(pred_diag.get('r2_from_predictions'))}**, RMSE = **{_fmt_num(pred_diag.get('rmse_from_predictions'))}**, MAE = **{_fmt_num(pred_diag.get('mae_from_predictions'))}**.")
        if pred_diag.get("split_counts"):
            md.append(f"Split counts: **{pred_diag['split_counts']}**.")
        if config.include_predictions_preview:
            md.append("")
            md.append(_df_preview_markdown(predictions, config.max_preview_rows))
        sections.append({"section": "Prediction diagnostics", "status": "created", "rows": len(predictions), "notes": "Prediction table preview and detected columns."})

    md.append("")
    md.append("## 5. Applicability domain")
    md.append("")
    if ad_summary is None or ad_summary.empty:
        md.append("No applicability-domain table was supplied. For a publication-ready QSAR report, include AD coverage, out-of-domain compounds, and nearest-neighbour or leverage evidence.")
        sections.append({"section": "Applicability domain", "status": "missing", "rows": 0, "notes": "No applicability-domain summary supplied."})
    else:
        if ad_prof.get("coverage") is not None:
            cov = float(ad_prof["coverage"])
            cov_pct = cov * 100 if cov <= 1.0 else cov
            md.append(f"Estimated AD coverage: **{cov_pct:.1f}%**.")
        if ad_prof.get("out_of_domain_count") is not None:
            md.append(f"Out-of-domain compounds detected: **{ad_prof['out_of_domain_count']}**.")
        md.append(_df_preview_markdown(ad_summary, config.max_preview_rows))
        sections.append({"section": "Applicability domain", "status": "created", "rows": len(ad_summary), "notes": "AD summary and coverage interpretation."})

    md.append("")
    md.append("## 6. Interpretation and chemical rationale")
    md.append("")
    if explanation_summary is None or explanation_summary.empty:
        md.append("No explanation table was supplied. Add descriptor importance, coefficients, SHAP values or fragment contributions to support chemical interpretation.")
        sections.append({"section": "Interpretation and chemical rationale", "status": "missing", "rows": 0, "notes": "No explanation table supplied."})
    else:
        if expl_prof.get("top_feature_pairs"):
            md.append("Most influential features/descriptors detected:")
            for name, value in expl_prof["top_feature_pairs"][:8]:
                md.append(f"- **{name}**: {_fmt_num(value)}")
        else:
            tops = ", ".join(expl_prof.get("top_features", [])[:8])
            if tops:
                md.append(f"Detected explanatory features: **{tops}**.")
        md.append("")
        md.append(_df_preview_markdown(explanation_summary, config.max_preview_rows))
        sections.append({"section": "Interpretation and chemical rationale", "status": "created", "rows": len(explanation_summary), "notes": "Feature/explanation summary."})

    md.append("")
    md.append("## 7. Validation dashboard details")
    md.append("")
    if validation_summary is None or validation_summary.empty:
        md.append("No validation-summary table was supplied.")
        sections.append({"section": "Validation dashboard details", "status": "missing", "rows": 0, "notes": "No validation summary supplied."})
    else:
        md.append(_df_preview_markdown(validation_summary, config.max_preview_rows))
        sections.append({"section": "Validation dashboard details", "status": "created", "rows": len(validation_summary), "notes": "Validation summary from QSAR Validation Dashboard."})

    if config.include_oecd_readiness:
        md.append("")
        md.append("## 8. OECD QSAR readiness check")
        md.append("")
        oecd = _oecd_df(ds_prof, metrics_lookup, ad_prof, expl_prof)
        md.append(_df_preview_markdown(oecd, config.max_preview_rows))
        sections.append({"section": "OECD QSAR readiness check", "status": "created", "rows": len(oecd), "notes": "OECD-style readiness evidence table."})

    md.append("")
    md.append("## 9. Report quality flags")
    md.append("")
    flags = _quality_flags(ds_prof, pred_diag, ad_prof, expl_prof)
    md.append(_df_preview_markdown(_flags_to_df(flags), config.max_preview_rows))
    sections.append({"section": "Report quality flags", "status": "created", "rows": len(flags), "notes": "Actionable report-completeness flags."})

    if config.include_auto_conclusions:
        md.append("")
        md.append("## 10. Auto-generated conclusions")
        md.append("")
        conclusion_items = []
        r2 = _first_metric(metrics_lookup, ["r2", "q2"], ["test", "external", "validation", ""])
        if r2 is None and pred_diag.get("r2_from_predictions") is not None:
            r2 = float(pred_diag["r2_from_predictions"])
        if r2 is not None:
            conclusion_items.append(f"The model shows **{_classify_r2(r2)}** predictive behaviour based on the available R²/Q²-like metric ({_fmt_num(r2)}).")
        if ad_prof.get("coverage") is None:
            conclusion_items.append("The report is not yet complete for serious external prediction because applicability-domain evidence is missing.")
        if not expl_prof:
            conclusion_items.append("Chemical interpretation remains incomplete because no explanation or descriptor-importance table was supplied.")
        if dataset_shape[0] and dataset_shape[0] < 30:
            conclusion_items.append("The dataset is small; conclusions should be considered preliminary and should preferably be supported by external validation or repeated resampling.")
        if not conclusion_items:
            conclusion_items.append("The supplied tables support a coherent QSAR report; final manuscript interpretation should still check endpoint consistency, AD evidence and mechanistic plausibility.")
        for item in conclusion_items:
            md.append(f"- {item}")
        sections.append({"section": "Auto-generated conclusions", "status": "created", "rows": len(conclusion_items), "notes": "Narrative report conclusion draft."})

    if config.include_publication_checklist:
        md.append("")
        md.append("## 11. Publication checklist")
        md.append("")
        checklist = [
            "Endpoint definition and units are explicitly stated.",
            "Molecular standardization and duplicate handling are documented.",
            "Descriptor/fingerprint calculation settings are recorded.",
            "Training/test/external split or cross-validation strategy is documented.",
            "Applicability-domain method and thresholds are reported.",
            "Model hyperparameters and software versions are recorded.",
            "Interpretability evidence is included, preferably with chemically meaningful descriptors/fragments.",
            "Outliers and out-of-domain predictions are listed or exported.",
        ]
        for item in checklist:
            md.append(f"- [ ] {item}")
        sections.append({"section": "Publication checklist", "status": "created", "rows": len(checklist), "notes": "Manual checklist for manuscript/report preparation."})

    if config.include_limitations:
        md.append("")
        md.append("## 12. Limitations and responsible use")
        md.append("")
        md.append("QSAR predictions should not be interpreted as experimental facts. They are model-based estimates and depend on dataset quality, molecular standardization, descriptor choice, validation strategy, and applicability domain. Out-of-domain predictions should be flagged and treated as lower-confidence hypotheses.")
        sections.append({"section": "Limitations", "status": "created", "rows": 1, "notes": "Responsible-use statement."})

    markdown = "\n".join(md) + "\n"
    html_body = _simple_markdown_to_html(markdown)
    html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>{_html.escape(config.title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; line-height: 1.5; color: #172033; background: #ffffff; }}
h1 {{ color: #0f3b5f; border-bottom: 4px solid #14b8a6; padding-bottom: .4rem; font-size: 2rem; }}
h2 {{ color: #0f3b5f; margin-top: 1.8rem; font-size: 1.35rem; }}
h3 {{ color: #31546c; margin-top: 1.2rem; }}
p, li {{ font-size: .98rem; }}
strong {{ color: #0f172a; }}
code, pre {{ background: #f3f4f6; padding: .1rem .25rem; border-radius: .25rem; }}
pre {{ padding: .75rem; overflow-x: auto; }}
ul {{ padding-left: 1.4rem; }}
.report-table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; font-size: .92rem; }}
.report-table th {{ background: #eef6f8; color: #0f3b5f; text-align: left; }}
.report-table th, .report-table td {{ border: 1px solid #d1d5db; padding: .38rem .55rem; vertical-align: top; }}
.report-table tr:nth-child(even) td {{ background: #f8fafc; }}
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
        "detected_target_column": ds_prof.get("target_column"),
        "detected_smiles_column": ds_prof.get("smiles_column"),
        "ad_coverage": ad_prof.get("coverage"),
        "r2_from_predictions": pred_diag.get("r2_from_predictions"),
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
