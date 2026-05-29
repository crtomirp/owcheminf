from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QSARDescriptorExplorerConfig:
    """Configuration for descriptor quality/redundancy exploration."""

    target_column: str = ""
    id_column: str = "compound_id"
    missing_threshold: float = 0.20
    low_variance_threshold: float = 1.0e-12
    high_correlation_threshold: float = 0.95
    max_correlation_pairs: int = 500
    exclude_columns: tuple[str, ...] = (
        "smiles",
        "canonical_smiles",
        "inchi",
        "inchikey",
        "compound_id",
        "name",
        "split",
        "set",
        "source",
    )


@dataclass
class QSARDescriptorExplorerResult:
    descriptor_summary: pd.DataFrame
    category_summary: pd.DataFrame
    correlation_pairs: pd.DataFrame
    filtered_data: pd.DataFrame
    quality_report: pd.DataFrame
    html_report: str
    markdown_report: str


_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("constitutional", ("molwt", "exactmolwt", "heavyatom", "numatom", "numvalence", "formula", "bertzct")),
    ("physicochemical", ("logp", "mr", "tpsa", "labuteasa", "qed", "fractioncsp3", "lipinski", "crippen")),
    ("hydrogen-bonding", ("hbd", "hba", "donor", "acceptor", "numhdonors", "numhacceptors")),
    ("charge/electronic", ("charge", "estate", "peoe", "vsa_estate", "bcut", "chi", "kappa")),
    ("topological", ("balaban", "hallkier", "kier", "ipc", "diameter", "radius", "topo", "connectivity")),
    ("fragment/count", ("fr_", "numaromatic", "numaliphatic", "numrotatable", "ring", "hetero", "nhoh", "no_count")),
    ("fingerprint", ("fp", "fingerprint", "maccs", "morgan", "ecfp", "fcfp", "bit_", "pubchem")),
    ("3d/geometric", ("pmi", "npr", "asphericity", "eccentricity", "inertial", "radiusofgyration", "spherocity", "pbf")),
)


def infer_descriptor_category(name: str) -> str:
    low = str(name).strip().lower()
    for category, keys in _CATEGORY_RULES:
        if any(k in low for k in keys):
            return category
    return "other/unknown"


def _numeric_descriptor_columns(df: pd.DataFrame, config: QSARDescriptorExplorerConfig) -> list[str]:
    excluded = {c.strip().lower() for c in config.exclude_columns if c}
    if config.target_column:
        excluded.add(config.target_column.strip().lower())
    if config.id_column:
        excluded.add(config.id_column.strip().lower())
    cols: list[str] = []
    for col in df.columns:
        if str(col).strip().lower() in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _safe_float(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def explore_qsar_descriptors(
    data: pd.DataFrame,
    config: QSARDescriptorExplorerConfig | None = None,
) -> QSARDescriptorExplorerResult:
    """Analyze descriptor matrix quality for QSAR/QSPR modeling.

    The function intentionally depends only on pandas/numpy so it can be tested
    outside Orange and reused from CLI/report generators.
    """

    config = config or QSARDescriptorExplorerConfig()
    if data is None or data.empty:
        empty = pd.DataFrame()
        report = _quality_report(0, 0, 0, 0, 0, 0, 0)
        html = _html_report(report, empty, empty, empty, config)
        md = _markdown_report(report, empty, empty, empty, config)
        return QSARDescriptorExplorerResult(empty, empty, empty, empty, report, html, md)

    df = data.copy()
    descriptor_cols = _numeric_descriptor_columns(df, config)
    n_rows = len(df)

    summaries: list[dict[str, object]] = []
    for col in descriptor_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        n_missing = int(s.isna().sum())
        missing_fraction = n_missing / max(n_rows, 1)
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()
        nunique = int(finite.nunique(dropna=True)) if len(finite) else 0
        variance = _safe_float(finite.var(ddof=0), 0.0) if len(finite) else 0.0
        std = _safe_float(finite.std(ddof=0), 0.0) if len(finite) else 0.0
        status_flags: list[str] = []
        if missing_fraction > config.missing_threshold:
            status_flags.append("high_missing")
        if nunique <= 1 or variance <= config.low_variance_threshold:
            status_flags.append("low_variance")
        if not status_flags:
            status_flags.append("ok")
        summaries.append(
            {
                "descriptor": col,
                "category": infer_descriptor_category(col),
                "n": n_rows,
                "missing": n_missing,
                "missing_fraction": missing_fraction,
                "n_unique": nunique,
                "mean": _safe_float(finite.mean()) if len(finite) else np.nan,
                "std": std,
                "variance": variance,
                "min": _safe_float(finite.min()) if len(finite) else np.nan,
                "median": _safe_float(finite.median()) if len(finite) else np.nan,
                "max": _safe_float(finite.max()) if len(finite) else np.nan,
                "status": ";".join(status_flags),
            }
        )

    descriptor_summary = pd.DataFrame(summaries)
    usable_cols = []
    if not descriptor_summary.empty:
        bad_status = descriptor_summary["status"].astype(str).str.contains("high_missing|low_variance", regex=True)
        usable_cols = descriptor_summary.loc[~bad_status, "descriptor"].astype(str).tolist()

    category_summary = _category_summary(descriptor_summary)
    correlation_pairs = _correlation_pairs(df, usable_cols, config)
    redundant = set(correlation_pairs.get("descriptor_b", pd.Series(dtype=str)).astype(str).tolist())

    keep_descriptor_cols = [c for c in usable_cols if c not in redundant]
    keep_non_descriptor_cols = [c for c in df.columns if c not in descriptor_cols]
    filtered_data = df[keep_non_descriptor_cols + keep_descriptor_cols].copy()

    n_high_missing = int(descriptor_summary["status"].astype(str).str.contains("high_missing").sum()) if not descriptor_summary.empty else 0
    n_low_variance = int(descriptor_summary["status"].astype(str).str.contains("low_variance").sum()) if not descriptor_summary.empty else 0
    n_redundant = len(redundant)
    report = _quality_report(
        n_rows,
        len(descriptor_cols),
        len(usable_cols),
        len(keep_descriptor_cols),
        n_high_missing,
        n_low_variance,
        n_redundant,
    )
    html = _html_report(report, descriptor_summary, category_summary, correlation_pairs, config)
    md = _markdown_report(report, descriptor_summary, category_summary, correlation_pairs, config)
    return QSARDescriptorExplorerResult(
        descriptor_summary=descriptor_summary,
        category_summary=category_summary,
        correlation_pairs=correlation_pairs,
        filtered_data=filtered_data,
        quality_report=report,
        html_report=html,
        markdown_report=md,
    )


def _category_summary(descriptor_summary: pd.DataFrame) -> pd.DataFrame:
    if descriptor_summary is None or descriptor_summary.empty:
        return pd.DataFrame(columns=["category", "descriptors", "ok", "high_missing", "low_variance", "mean_missing_fraction"])
    rows = []
    for category, g in descriptor_summary.groupby("category", dropna=False):
        status = g["status"].astype(str)
        rows.append(
            {
                "category": category,
                "descriptors": int(len(g)),
                "ok": int((status == "ok").sum()),
                "high_missing": int(status.str.contains("high_missing").sum()),
                "low_variance": int(status.str.contains("low_variance").sum()),
                "mean_missing_fraction": float(g["missing_fraction"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["descriptors", "category"], ascending=[False, True]).reset_index(drop=True)


def _correlation_pairs(df: pd.DataFrame, cols: Iterable[str], config: QSARDescriptorExplorerConfig) -> pd.DataFrame:
    cols = list(cols)
    if len(cols) < 2:
        return pd.DataFrame(columns=["descriptor_a", "descriptor_b", "abs_correlation", "correlation"])
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = X.corr(method="pearson", min_periods=max(3, min(10, len(df) // 3)))
    rows: list[dict[str, object]] = []
    vals = corr.to_numpy(dtype=float)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = vals[i, j]
            if not np.isfinite(r):
                continue
            ar = abs(float(r))
            if ar >= config.high_correlation_threshold:
                rows.append(
                    {
                        "descriptor_a": cols[i],
                        "descriptor_b": cols[j],
                        "abs_correlation": ar,
                        "correlation": float(r),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["descriptor_a", "descriptor_b", "abs_correlation", "correlation"])
    return out.sort_values("abs_correlation", ascending=False).head(int(config.max_correlation_pairs)).reset_index(drop=True)


def _quality_report(n_rows, n_desc, n_usable, n_final, n_high_missing, n_low_variance, n_redundant) -> pd.DataFrame:
    rows = [
        ("records", n_rows, "Input rows."),
        ("numeric_descriptor_candidates", n_desc, "Numeric columns considered as descriptors after excluding ID/SMILES/target columns."),
        ("usable_after_basic_filters", n_usable, "Descriptors not flagged for missingness or low variance."),
        ("recommended_after_redundancy_filter", n_final, "Usable descriptors after removing one side of highly correlated pairs."),
        ("high_missing_descriptors", n_high_missing, "Descriptors above the configured missing-value threshold."),
        ("low_variance_descriptors", n_low_variance, "Constant or near-constant descriptors."),
        ("redundant_correlated_descriptors", n_redundant, "Descriptors suggested for removal due to high pairwise correlation."),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


def _fmt(x) -> str:
    try:
        f = float(x)
        if abs(f) >= 1000 or (abs(f) < 0.001 and f != 0):
            return f"{f:.3e}"
        return f"{f:.3f}"
    except Exception:
        return str(x)


def _html_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "<p><em>No records.</em></p>"
    show = df.head(max_rows).copy()
    return show.to_html(index=False, escape=True, classes="qsar-table", float_format=lambda x: f"{x:.4g}")


def _html_report(report, descriptor_summary, category_summary, correlation_pairs, config) -> str:
    metrics = {r.metric: r.value for r in report.itertuples(index=False)} if report is not None and not report.empty else {}
    n_desc = int(metrics.get("numeric_descriptor_candidates", 0) or 0)
    n_final = int(metrics.get("recommended_after_redundancy_filter", 0) or 0)
    n_bad = int(metrics.get("high_missing_descriptors", 0) or 0) + int(metrics.get("low_variance_descriptors", 0) or 0)
    decision = "Good descriptor matrix" if n_desc and n_bad == 0 else "Descriptor matrix needs curation" if n_desc else "No numeric descriptors detected"
    return f"""
    <html><head><style>
    body {{ font-family: Arial, sans-serif; color:#0f172a; background:#ffffff; }}
    h1 {{ font-size: 22px; margin-bottom: 4px; }}
    h2 {{ font-size: 17px; margin-top: 18px; border-bottom:1px solid #e2e8f0; padding-bottom:4px; }}
    .chip {{ display:inline-block; padding:4px 9px; border-radius:999px; background:#eff6ff; color:#1d4ed8; font-weight:600; }}
    .warn {{ background:#fff7ed; color:#c2410c; }}
    .grid {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:8px; }}
    .card {{ border:1px solid #e2e8f0; border-radius:10px; padding:10px; background:#f8fafc; }}
    .num {{ font-size:20px; font-weight:700; }}
    table.qsar-table {{ border-collapse:collapse; width:100%; font-size: 12px; }}
    table.qsar-table th {{ background:#f1f5f9; text-align:left; padding:5px; border:1px solid #e2e8f0; }}
    table.qsar-table td {{ padding:5px; border:1px solid #e2e8f0; }}
    </style></head><body>
    <h1>Descriptor Explorer Report</h1>
    <p><span class="chip {'warn' if n_bad else ''}">{decision}</span></p>
    <div class="grid">
      <div class="card"><div class="num">{metrics.get('records', 0)}</div><div>records</div></div>
      <div class="card"><div class="num">{n_desc}</div><div>numeric descriptor candidates</div></div>
      <div class="card"><div class="num">{metrics.get('usable_after_basic_filters', 0)}</div><div>usable after basic filters</div></div>
      <div class="card"><div class="num">{n_final}</div><div>recommended final descriptors</div></div>
    </div>
    <h2>Quality flags</h2>
    {_html_table(report, 20)}
    <h2>Descriptor categories</h2>
    {_html_table(category_summary, 20)}
    <h2>Highest-risk descriptors</h2>
    {_html_table(_risk_descriptors(descriptor_summary), 20)}
    <h2>Highly correlated pairs</h2>
    <p>Threshold: |r| ≥ {_fmt(config.high_correlation_threshold)}. These pairs indicate redundant information; remove one descriptor from each pair before final modeling.</p>
    {_html_table(correlation_pairs, 20)}
    <h2>Recommended workflow use</h2>
    <p>Use the filtered output as input for Descriptor Preselector or QSAR/QSPR Model Hub. Keep the full descriptor summary for the final QSAR report and publication checklist.</p>
    </body></html>
    """


def _risk_descriptors(descriptor_summary: pd.DataFrame) -> pd.DataFrame:
    if descriptor_summary is None or descriptor_summary.empty:
        return pd.DataFrame()
    df = descriptor_summary.copy()
    df["risk_score"] = df["missing_fraction"].astype(float) + (df["status"].astype(str).str.contains("low_variance").astype(float))
    cols = ["descriptor", "category", "missing_fraction", "n_unique", "variance", "status"]
    return df.sort_values(["risk_score", "missing_fraction"], ascending=False)[cols]


def _markdown_report(report, descriptor_summary, category_summary, correlation_pairs, config) -> str:
    metrics = {str(r.metric): r.value for r in report.itertuples(index=False)} if report is not None and not report.empty else {}
    lines = [
        "# Descriptor Explorer Report",
        "",
        f"Records: {metrics.get('records', 0)}",
        f"Numeric descriptor candidates: {metrics.get('numeric_descriptor_candidates', 0)}",
        f"Usable after basic filters: {metrics.get('usable_after_basic_filters', 0)}",
        f"Recommended after redundancy filter: {metrics.get('recommended_after_redundancy_filter', 0)}",
        "",
        "## Interpretation",
    ]
    if int(metrics.get("numeric_descriptor_candidates", 0) or 0) == 0:
        lines.append("No numeric descriptor columns were detected. Check whether the descriptor generator output is connected.")
    else:
        lines.append("Descriptors were screened for missingness, low variance and high pairwise Pearson correlation.")
        lines.append(f"Correlation threshold used: |r| >= {config.high_correlation_threshold}.")
    return "\n".join(lines)
