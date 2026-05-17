"""Descriptor pre-selection: missing-value, variance, and correlation filters.

The filters are intentionally applied in strict, memory-friendly stages:

1. Drop fully empty descriptors first.
2. Drop descriptors above the configured missing-value threshold.
3. Compute variance only on the reduced matrix and drop low-variance columns.
4. Compute correlation only on the remaining descriptors.

This order avoids sklearn/pandas warnings from all-NaN columns and prevents
large unnecessary correlation matrices.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class DescriptorFilterConfig:
    max_missing_fraction: float = 0.20
    min_variance: float = 0.01
    max_correlation: float = 0.90
    correlation_method: str = "pearson"
    target_column: str = ""
    id_columns: tuple[str, ...] = ()
    impute_for_stats: bool = True
    # Above this number of features, use a memory-safe greedy correlation filter
    # instead of building a full n_features × n_features matrix.
    max_correlation_features: int = 1500
    correlation_block_size: int = 256
    # Hard cap before pairwise correlation. If more features remain after
    # variance filtering, keep only the strongest features by target-correlation
    # score (or variance if no target is selected). This prevents very large
    # O(p²) correlation workloads from killing Orange. Set <=0 to disable.
    max_features_before_correlation: int = 3000
    # Final cap after all filters. This is a safety valve for downstream QSAR
    # widgets: very wide descriptor tables can still be expensive even if they
    # survived missing/variance/correlation filters. Set <=0 to disable.
    max_output_features: int = 1000
    # Write cleaned numeric descriptor values to the output table.
    # This removes Mordred/Orange "?" values from kept descriptor columns.
    impute_output_descriptors: bool = True
    output_impute_strategy: str = "median"  # "median", "mean", or "zero"
    # Phase 3.8: descriptor quality scoring. The score is reported but does
    # not remove features by default; it helps diagnose bad Mordred families.
    enable_quality_score: bool = True


@dataclass
class DescriptorFilterResult:
    kept_features: list[str]
    removed_empty: list[str]
    removed_missing: list[str]
    removed_low_variance: list[str]
    removed_pre_correlation_cap: list[str]
    removed_correlated: list[str]
    removed_final_cap: list[str]
    corr_clusters: list[dict]
    variance_series: pd.Series
    missing_series: pd.Series
    n_input: int
    n_after_empty: int
    n_after_missing: int
    n_after_variance: int
    n_after_pre_correlation_cap: int
    n_after_correlation: int
    n_output: int
    report_df: pd.DataFrame
    notes: list[str]
    quality_summary: dict[str, object] | None = None


def _as_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a numeric-only copy with infinities converted to NaN."""
    if not columns:
        return pd.DataFrame(index=df.index)
    X = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def _impute_for_stats(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute a reduced descriptor matrix without sklearn warnings."""
    if X.empty:
        return X.copy()
    med = X.median(axis=0, skipna=True)
    return X.fillna(med).fillna(0.0)


def _safe_feature_scores(
    df: pd.DataFrame,
    X_stats: pd.DataFrame,
    variance: pd.Series,
    target_column: str,
) -> dict[str, float]:
    """Score features for representative selection in correlated groups.

    If a numeric target is available, use absolute feature-target correlation.
    Otherwise use descriptor variance. All invalid scores fall back to 0.
    """
    feat_cols = list(X_stats.columns)
    if not feat_cols:
        return {}
    if target_column and target_column in df.columns:
        y = pd.to_numeric(df[target_column], errors="coerce")
        if y.notna().sum() > 2:
            out: dict[str, float] = {}
            for col in feat_cols:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    val = pd.Series(X_stats[col]).corr(y)
                out[col] = 0.0 if not np.isfinite(val) else abs(float(val))
            return out
    return {col: float(variance.get(col, 0.0)) if np.isfinite(float(variance.get(col, 0.0))) else 0.0 for col in feat_cols}




def _descriptor_family(name: str) -> str:
    """Return a compact descriptor family label for diagnostics.

    Mordred descriptors often share long prefixes (for example MINS..., MAX...,
    ATS..., AATS...). Grouping by a stable prefix makes it easier to see which
    descriptor families are mostly undefined or removed. The function is purely
    diagnostic and never changes filtering decisions.
    """
    text = str(name or "").strip()
    if not text:
        return "unknown"
    # Prefer the leading alphabetic/underscore prefix until the first digit.
    import re
    m = re.match(r"^([A-Za-z_]+)", text)
    if m:
        family = m.group(1).strip("_")
        return family[:32] or "unknown"
    return text.split("_")[0][:32] or "unknown"


def _safe_minmax(values: pd.Series) -> pd.Series:
    """Scale finite numeric values to 0..1; degenerate series becomes zeros."""
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if v.empty:
        return v
    lo = float(v.min())
    hi = float(v.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
    return (v - lo) / (hi - lo)


def _quality_summary_from_report(report_df: pd.DataFrame) -> dict[str, object]:
    """Summarise descriptor quality/family problems for the widget summary."""
    if report_df.empty or "family" not in report_df.columns:
        return {"top_removed_families": [], "mean_quality_kept": None, "mean_quality_all": None}
    removed = report_df[report_df["status"].astype(str) != "kept"]
    top = []
    if not removed.empty:
        grp = removed.groupby("family").size().sort_values(ascending=False).head(10)
        top = [(str(k), int(v)) for k, v in grp.items()]
    kept = report_df[report_df["status"].astype(str) == "kept"]
    q_all = pd.to_numeric(report_df.get("quality_score", pd.Series(dtype=float)), errors="coerce")
    q_kept = pd.to_numeric(kept.get("quality_score", pd.Series(dtype=float)), errors="coerce") if not kept.empty else pd.Series(dtype=float)
    return {
        "top_removed_families": top,
        "mean_quality_kept": None if q_kept.dropna().empty else round(float(q_kept.mean()), 2),
        "mean_quality_all": None if q_all.dropna().empty else round(float(q_all.mean()), 2),
    }

def _cap_features_before_correlation(
    feat_cols: list[str],
    feature_scores: dict[str, float],
    variance: pd.Series,
    max_features: int,
) -> tuple[list[str], list[str]]:
    """Limit features before the expensive correlation stage.

    The selected columns are the strongest by target-aware score where
    available; variance is used as a deterministic tie-breaker. The original
    table order is restored for kept features so downstream tables remain
    readable and predictable.
    """
    max_features = int(max_features or 0)
    if max_features <= 0 or len(feat_cols) <= max_features:
        return feat_cols, []

    ranked = sorted(
        feat_cols,
        key=lambda c: (
            float(feature_scores.get(c, 0.0) or 0.0),
            float(variance.get(c, 0.0) if np.isfinite(float(variance.get(c, 0.0) or 0.0)) else 0.0),
            c,
        ),
        reverse=True,
    )
    keep_set = set(ranked[:max_features])
    kept = [c for c in feat_cols if c in keep_set]
    removed = [c for c in feat_cols if c not in keep_set]
    return kept, removed


def _cap_output_features(
    feat_cols: list[str],
    feature_scores: dict[str, float],
    variance: pd.Series,
    max_features: int,
) -> tuple[list[str], list[str]]:
    """Apply a final target-aware cap after correlation filtering.

    This is deliberately deterministic. Selected features are ranked by the
    same feature score used for correlation representatives, then by variance,
    and final output order follows the original table order.
    """
    return _cap_features_before_correlation(
        feat_cols,
        feature_scores,
        variance,
        max_features=max_features,
    )



def _impute_output_descriptors(X: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Return descriptor values without NaN/inf for the final output table.

    Mordred descriptors often contain undefined values. Orange displays these
    as ``?``. After descriptors have survived the missing-value filter, keeping
    sparse ``?`` values is usually counterproductive for QSAR, correlation
    filters and downstream models. This function fills only the final kept
    descriptor columns; dropped descriptors remain dropped.
    """
    if X.empty:
        return X.copy()

    out = X.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    strategy = (strategy or "median").lower().strip()

    if strategy == "zero":
        return out.fillna(0.0)
    if strategy == "mean":
        fill_values = out.mean(axis=0, skipna=True)
    else:
        fill_values = out.median(axis=0, skipna=True)

    # A column that is still all-NaN should not occur after filtering, but this
    # fallback keeps the output safe even for pathological inputs.
    fill_values = fill_values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.fillna(fill_values).fillna(0.0)

def _standardized_float32(X: pd.DataFrame) -> np.ndarray:
    """Return row-wise matrix standardized per feature as writable float32.

    This keeps memory predictable for blockwise correlation checks.
    """
    arr = np.asarray(X.to_numpy(dtype=np.float32, copy=True), dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape((arr.shape[0], -1))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    arr -= arr.mean(axis=0, keepdims=True, dtype=np.float32)
    sd = arr.std(axis=0, keepdims=True, dtype=np.float32)
    sd[sd == 0] = 1.0
    arr /= sd
    return arr


def _greedy_blockwise_correlation_filter(
    X_stats: pd.DataFrame,
    feature_scores: dict[str, float],
    threshold: float,
    block_size: int = 256,
) -> tuple[list[str], list[str], list[dict]]:
    """Memory-safe high-correlation filter.

    It avoids allocating a full n_features × n_features matrix. Features are
    considered from strongest to weakest score. A feature is kept only if its
    absolute correlation with all already-kept features is below threshold.
    """
    feat_cols = list(X_stats.columns)
    if len(feat_cols) < 2 or threshold >= 1.0:
        return feat_cols, [], []

    arr = _standardized_float32(X_stats)
    n_rows = max(int(arr.shape[0]), 1)
    denom = float(max(n_rows - 1, 1))
    order = sorted(
        range(len(feat_cols)),
        key=lambda i: (feature_scores.get(feat_cols[i], 0.0), feat_cols[i]),
        reverse=True,
    )

    kept_idx: list[int] = []
    removed_idx: list[int] = []
    cluster_by_kept: dict[int, dict] = {}
    block_size = max(32, int(block_size or 256))

    for idx in order:
        if not kept_idx:
            kept_idx.append(idx)
            continue

        x = arr[:, idx]
        max_abs_r = 0.0
        best_kept = None
        # Check against kept features in blocks to bound temporary memory.
        for start in range(0, len(kept_idx), block_size):
            block = kept_idx[start:start + block_size]
            kb = arr[:, block]
            r = (kb.T @ x) / denom
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            abs_r = np.abs(r)
            j = int(abs_r.argmax()) if abs_r.size else 0
            if abs_r.size and float(abs_r[j]) > max_abs_r:
                max_abs_r = float(abs_r[j])
                best_kept = block[j]
            if max_abs_r >= threshold:
                break

        if max_abs_r >= threshold and best_kept is not None:
            removed_idx.append(idx)
            rec = cluster_by_kept.setdefault(best_kept, {"kept": feat_cols[best_kept], "removed": [], "max_r": 0.0, "size": 1})
            rec["removed"].append(feat_cols[idx])
            rec["max_r"] = max(float(rec["max_r"]), round(max_abs_r, 4))
            rec["size"] = int(rec["size"]) + 1
        else:
            kept_idx.append(idx)

    kept_set = {feat_cols[i] for i in kept_idx}
    kept_ordered = [c for c in feat_cols if c in kept_set]
    removed = [feat_cols[i] for i in removed_idx]
    clusters = list(cluster_by_kept.values())
    return kept_ordered, sorted(removed), clusters


def run_descriptor_filter(
    df: pd.DataFrame,
    config: DescriptorFilterConfig,
) -> tuple[pd.DataFrame, DescriptorFilterResult]:
    """Apply staged descriptor filters and return ``(filtered_df, result)``.

    Pass-through columns (SMILES, inchikey, target, non-numeric columns and
    explicit id_columns) are never removed. Only numeric descriptor columns are
    filtered.
    """
    passthrough = set(config.id_columns)
    if config.target_column:
        passthrough.add(config.target_column)

    passthrough |= {c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])}

    all_features = [
        c for c in df.columns
        if c not in passthrough and pd.api.types.is_numeric_dtype(df[c])
    ]
    n_input = len(all_features)
    notes: list[str] = []

    X0 = _as_numeric_frame(df, all_features)

    # Step 1: all-empty descriptors. These must be removed before any variance,
    # imputation or correlation step, otherwise pandas/sklearn emit warnings and
    # may allocate useless matrices.
    missing_all = X0.isna().all(axis=0) if not X0.empty else pd.Series(dtype=bool)
    removed_empty = missing_all[missing_all].index.tolist()
    X1 = X0.drop(columns=removed_empty, errors="ignore")
    n_after_empty = X1.shape[1]

    # Step 2: remaining high-missing descriptors.
    missing_frac = X1.isna().mean(axis=0) if not X1.empty else pd.Series(dtype=float)
    removed_missing = missing_frac[missing_frac > config.max_missing_fraction].index.tolist()
    X2 = X1.drop(columns=removed_missing, errors="ignore")
    n_after_missing = X2.shape[1]

    # Step 3: variance filter on the already reduced matrix.
    if config.impute_for_stats:
        X2_stats = _impute_for_stats(X2)
    else:
        X2_stats = X2.dropna(axis=0).copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        variance = X2_stats.var(axis=0, ddof=0) if not X2_stats.empty else pd.Series(dtype=float)
    variance = variance.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    removed_low_var = variance[variance < config.min_variance].index.tolist()
    X3 = X2.drop(columns=removed_low_var, errors="ignore")
    X3_stats = X2_stats.drop(columns=removed_low_var, errors="ignore")
    feat_cols = list(X3.columns)
    n_after_variance = len(feat_cols)

    # Step 4a: optional hard cap before correlation. This is deliberately
    # before the pairwise correlation step because correlation is the only
    # O(p²)-style part of the workflow. It keeps Orange responsive on large
    # descriptor matrices.
    feature_scores = _safe_feature_scores(df, X3_stats, variance, config.target_column)
    feat_cols, removed_pre_corr_cap = _cap_features_before_correlation(
        feat_cols,
        feature_scores,
        variance,
        max_features=int(config.max_features_before_correlation),
    )
    if removed_pre_corr_cap:
        notes.append(
            f"Pre-correlation cap kept {len(feat_cols)} strongest features and "
            f"removed {len(removed_pre_corr_cap)} lower-ranked features before "
            f"correlation filtering. Cap={config.max_features_before_correlation}."
        )
        X3_stats = X3_stats.loc[:, feat_cols]
    n_after_pre_corr_cap = len(feat_cols)

    # Step 4b: correlation filter on the remaining descriptors only.
    removed_correlated: list[str] = []
    corr_clusters: list[dict] = []

    if len(feat_cols) >= 2 and config.max_correlation < 1.0:
        if len(feat_cols) > int(config.max_correlation_features):
            notes.append(
                f"Correlation filter used memory-safe greedy mode because {len(feat_cols)} "
                f"features remained after variance filtering; full-matrix limit is "
                f"{config.max_correlation_features}."
            )
            kept_corr, removed_correlated, corr_clusters = _greedy_blockwise_correlation_filter(
                X3_stats.loc[:, feat_cols],
                feature_scores,
                threshold=float(config.max_correlation),
                block_size=int(config.correlation_block_size),
            )
            feat_cols = kept_corr
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if config.correlation_method == "spearman":
                    corr_mat = X3_stats.rank(method="average").corr()
                else:
                    corr_mat = X3_stats.corr()

            corr_arr = np.array(corr_mat.to_numpy(dtype=np.float32), dtype=np.float32, copy=True)
            corr_arr = np.nan_to_num(corr_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            np.fill_diagonal(corr_arr, 1.0)
            dist_arr = (1.0 - np.abs(corr_arr)).astype(np.float32, copy=False)
            np.maximum(dist_arr, 0.0, out=dist_arr)
            np.fill_diagonal(dist_arr, 0.0)

            condensed = squareform(dist_arr, checks=False)
            Z = linkage(condensed, method="average")
            labels = fcluster(Z, t=1.0 - config.max_correlation, criterion="distance")

            target_corr = feature_scores

            cluster_map: dict[int, list[str]] = {}
            for col, lbl in zip(feat_cols, labels):
                cluster_map.setdefault(int(lbl), []).append(col)

            col_pos = {c: i for i, c in enumerate(feat_cols)}
            to_remove: set[str] = set()
            for members in cluster_map.values():
                if len(members) == 1:
                    continue
                rep = max(members, key=lambda c: target_corr.get(c, 0.0))
                rest = [c for c in members if c != rep]
                rep_i = col_pos[rep]
                max_r = 0.0
                for c in rest:
                    max_r = max(max_r, abs(float(corr_arr[rep_i, col_pos[c]])))
                corr_clusters.append({"kept": rep, "removed": rest, "max_r": round(max_r, 4), "size": len(members)})
                to_remove.update(rest)

            removed_correlated = sorted(to_remove)
            feat_cols = [c for c in feat_cols if c not in to_remove]

    n_after_correlation = len(feat_cols)

    # Step 5: optional final cap for downstream modeling. This keeps QSAR
    # widgets responsive when thousands of descriptors survive correlation.
    feat_cols, removed_final_cap = _cap_output_features(
        feat_cols,
        feature_scores,
        variance,
        max_features=int(config.max_output_features),
    )
    if removed_final_cap:
        notes.append(
            f"Final output cap kept {len(feat_cols)} strongest features and "
            f"removed {len(removed_final_cap)} additional features after correlation. "
            f"Cap={config.max_output_features}."
        )

    # Build report for all original numeric descriptor columns.
    reasons: dict[str, str] = {}
    for c in removed_empty:
        reasons[c] = "empty"
    for c in removed_missing:
        reasons[c] = "high_missing"
    for c in removed_low_var:
        reasons[c] = "low_variance"
    for c in removed_pre_corr_cap:
        reasons[c] = "pre_corr_cap"
    for c in removed_correlated:
        reasons[c] = "correlated"
    for c in removed_final_cap:
        reasons[c] = "final_cap"

    missing_report = X0.isna().mean(axis=0) if not X0.empty else pd.Series(dtype=float)

    # Phase 3.8: descriptor quality diagnostics. Missing values are penalised
    # strongly, variance adds information content, and target-aware score helps
    # rank descriptors when a numeric endpoint is available. This is reported
    # to the user but does not silently remove additional descriptors.
    variance_for_quality = pd.Series({c: float(variance.get(c, 0.0) or 0.0) for c in all_features})
    variance_scaled = _safe_minmax(np.log1p(variance_for_quality.clip(lower=0.0)))
    score_for_quality = pd.Series({c: float(feature_scores.get(c, 0.0) or 0.0) for c in all_features})
    score_scaled = _safe_minmax(score_for_quality)

    report_rows = []
    kept_set = set(feat_cols)
    for c in all_features:
        miss = float(missing_report.get(c, 0.0) or 0.0)
        status = reasons.get(c, "kept")
        # 0..100: clean, variable, and target-relevant descriptors score higher.
        quality = 100.0 * (
            0.55 * max(0.0, 1.0 - miss)
            + 0.25 * float(variance_scaled.get(c, 0.0) or 0.0)
            + 0.20 * float(score_scaled.get(c, 0.0) or 0.0)
        )
        # Removed features remain diagnostically useful but should not appear
        # deceptively high-quality if they failed a hard filter.
        if status != "kept":
            quality = min(quality, 49.0)
        imputed_values = 0
        if c in kept_set and c in X0.columns:
            imputed_values = int(X0[c].isna().sum())
        report_rows.append({
            "feature": c,
            "family": _descriptor_family(c),
            "missing_fraction": round(miss, 4),
            "missing_count": int(X0[c].isna().sum()) if c in X0.columns else 0,
            "variance": round(float(variance.get(c, np.nan)) if c not in removed_empty + removed_missing else np.nan, 6),
            "selection_score": round(float(feature_scores.get(c, 0.0) or 0.0), 6),
            "quality_score": round(float(quality), 2),
            "imputed_output_values": imputed_values,
            "status": status,
        })
    report_df = pd.DataFrame(report_rows)
    quality_summary = _quality_summary_from_report(report_df)

    out_cols = list(passthrough.intersection(df.columns)) + feat_cols
    out_cols_ordered = [c for c in df.columns if c in set(out_cols)]
    filtered_df = df.loc[:, out_cols_ordered].copy()

    # Critical Mordred cleanup: Orange displays NaN values as ``?``. For the
    # final kept descriptor columns, replace remaining missing values with
    # deterministic per-column statistics so downstream modeling receives a
    # fully numeric matrix. Pass-through columns such as SMILES, inchikey and
    # target values are not modified here.
    if config.impute_output_descriptors and feat_cols:
        kept_present = [c for c in feat_cols if c in filtered_df.columns]
        filtered_df.loc[:, kept_present] = _impute_output_descriptors(
            _as_numeric_frame(filtered_df, kept_present),
            strategy=config.output_impute_strategy,
        )
        if kept_present:
            n_imputed = int(_as_numeric_frame(df, kept_present).isna().sum().sum())
            notes.append(
                f"Output descriptor matrix was imputed with strategy="
                f"'{config.output_impute_strategy}' to remove remaining Mordred/Orange '?' values "
                f"({n_imputed} cells filled in kept descriptors)."
            )

    if config.enable_quality_score and quality_summary:
        mq = quality_summary.get("mean_quality_kept")
        if mq is not None:
            notes.append(f"Mean quality score of kept descriptors: {mq}/100.")
        top_fam = quality_summary.get("top_removed_families") or []
        if top_fam:
            fam_txt = ", ".join(f"{name} ({count})" for name, count in top_fam[:5])
            notes.append(f"Top removed descriptor families: {fam_txt}.")

    return filtered_df, DescriptorFilterResult(
        kept_features=feat_cols,
        removed_empty=removed_empty,
        removed_missing=removed_missing,
        removed_low_variance=removed_low_var,
        removed_pre_correlation_cap=removed_pre_corr_cap,
        removed_correlated=removed_correlated,
        removed_final_cap=removed_final_cap,
        corr_clusters=corr_clusters,
        variance_series=variance,
        missing_series=missing_report,
        n_input=n_input,
        n_after_empty=n_after_empty,
        n_after_missing=n_after_missing,
        n_after_variance=n_after_variance,
        n_after_pre_correlation_cap=n_after_pre_corr_cap,
        n_after_correlation=n_after_correlation,
        n_output=len(feat_cols),
        report_df=report_df,
        notes=notes,
        quality_summary=quality_summary,
    )
