from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from chem_inf_widgets.chemcore.qsar.mlr_selection import ADConfig
from chem_inf_widgets.chemcore.services.applicability_domain_service import (
    fit_applicability_domain,
    score_applicability_domain,
)


DEFAULT_EXCLUDED_COLUMNS = {
    "compound_id", "molecule_id", "id", "name", "smiles", "canonical_smiles",
    "standardized_smiles", "inchi", "inchikey", "standard_inchikey",
    "pactivity", "activity", "standard_value", "standard_units", "standard_type",
    "observed", "predicted", "residual", "abs_residual", "split", "class", "label",
}


@dataclass(frozen=True)
class ADWorkbenchConfig:
    id_column: str = "compound_id"
    combine_mode: str = "and"
    use_williams: bool = True
    use_knn: bool = True
    use_mahalanobis: bool = False
    knn_k: int = 5
    knn_quantile: float = 0.95
    maha_alpha: float = 0.95
    maha_use_chi2: bool = True
    min_non_missing_fraction: float = 0.70
    drop_constant_features: bool = True
    feature_columns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ADWorkbenchResult:
    scored_query: pd.DataFrame
    scored_reference: pd.DataFrame
    out_of_domain: pd.DataFrame
    summary: pd.DataFrame
    method_details: pd.DataFrame
    feature_names: list[str]
    summary_dict: dict[str, Any]


def _numeric_feature_columns(df: pd.DataFrame, cfg: ADWorkbenchConfig) -> list[str]:
    if cfg.feature_columns:
        missing = [c for c in cfg.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns are missing: {', '.join(missing)}")
        return list(cfg.feature_columns)
    keep: list[str] = []
    for col in df.columns:
        if str(col).lower() in DEFAULT_EXCLUDED_COLUMNS:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() < float(cfg.min_non_missing_fraction):
            continue
        if bool(cfg.drop_constant_features) and s.nunique(dropna=True) <= 1:
            continue
        if s.notna().sum() > 0:
            keep.append(str(col))
    return keep


def _common_features(ref_df: pd.DataFrame, query_df: pd.DataFrame, cfg: ADWorkbenchConfig) -> list[str]:
    ref_cols = _numeric_feature_columns(ref_df, cfg)
    if cfg.feature_columns:
        query_missing = [c for c in ref_cols if c not in query_df.columns]
        if query_missing:
            raise ValueError(f"Query data is missing requested feature columns: {', '.join(query_missing)}")
        return ref_cols
    query_cols = set(_numeric_feature_columns(query_df, cfg))
    common = [c for c in ref_cols if c in query_cols]
    if not common:
        raise ValueError("No shared numeric descriptor/fingerprint columns were detected.")
    return common


def _matrix(df: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    return df[list(feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _ad_cfg(cfg: ADWorkbenchConfig) -> ADConfig:
    return ADConfig(
        use_williams=bool(cfg.use_williams),
        use_knn=bool(cfg.use_knn),
        use_mahalanobis=bool(cfg.use_mahalanobis),
        combine_mode=str(cfg.combine_mode or "and"),
        knn_k=int(cfg.knn_k),
        knn_quantile=float(cfg.knn_quantile),
        maha_alpha=float(cfg.maha_alpha),
        maha_use_chi2=bool(cfg.maha_use_chi2),
    )


def _score_frame(df: pd.DataFrame, prediction, cfg: ADWorkbenchConfig, label: str) -> pd.DataFrame:
    out = df.copy()
    out["AD_dataset_role"] = label
    out["AD_leverage"] = prediction.leverage
    out["AD_h_star"] = prediction.h_star
    out["AD_in_williams"] = prediction.in_ad_williams.astype(bool)
    if prediction.knn_dist is not None:
        out["AD_knn_dist"] = prediction.knn_dist
        out["AD_knn_threshold"] = prediction.knn_threshold
    else:
        out["AD_knn_dist"] = np.nan
        out["AD_knn_threshold"] = np.nan
    out["AD_in_knn"] = prediction.in_ad_knn.astype(bool)
    if prediction.maha_d2 is not None:
        out["AD_maha_d2"] = prediction.maha_d2
        out["AD_maha_threshold"] = prediction.maha_threshold
    else:
        out["AD_maha_d2"] = np.nan
        out["AD_maha_threshold"] = np.nan
    out["AD_in_mahalanobis"] = prediction.in_ad_maha.astype(bool)
    out["AD_in_domain"] = prediction.in_ad.astype(bool)

    reasons: list[str] = []
    for i in range(len(out)):
        failed: list[str] = []
        if bool(cfg.use_williams) and not bool(prediction.in_ad_williams[i]):
            failed.append("leverage")
        if bool(cfg.use_knn) and not bool(prediction.in_ad_knn[i]):
            failed.append("knn")
        if bool(cfg.use_mahalanobis) and not bool(prediction.in_ad_maha[i]):
            failed.append("mahalanobis")
        reasons.append(";".join(failed) if failed else "in_domain")
    out["AD_reason"] = reasons
    return out


def evaluate_applicability_domain_workbench(
    reference_df: pd.DataFrame,
    query_df: pd.DataFrame | None = None,
    config: ADWorkbenchConfig = ADWorkbenchConfig(),
) -> ADWorkbenchResult:
    ref_df = reference_df.copy()
    q_df = ref_df.copy() if query_df is None else query_df.copy()
    feature_names = _common_features(ref_df, q_df, config)

    X_ref = _matrix(ref_df, feature_names)
    X_query = _matrix(q_df, feature_names)
    fit = fit_applicability_domain(X_ref, feature_names, ad_cfg=_ad_cfg(config))
    ref_pred = score_applicability_domain(fit, X_ref)
    query_pred = score_applicability_domain(fit, X_query)

    scored_ref = _score_frame(ref_df, ref_pred, config, "reference")
    scored_query = _score_frame(q_df, query_pred, config, "query")
    out_domain = scored_query[~scored_query["AD_in_domain"].astype(bool)].copy()

    summary_rows = [
        {"metric": "reference_rows", "value": int(len(scored_ref)), "note": "Rows used to fit AD"},
        {"metric": "query_rows", "value": int(len(scored_query)), "note": "Rows scored"},
        {"metric": "feature_count", "value": int(len(feature_names)), "note": "Shared numeric features"},
        {"metric": "query_in_domain", "value": int(scored_query["AD_in_domain"].sum()), "note": "Accepted query rows"},
        {"metric": "query_out_of_domain", "value": int(len(out_domain)), "note": "Rows requiring review"},
        {"metric": "h_star", "value": float(query_pred.h_star), "note": "Williams leverage threshold"},
    ]
    if query_pred.knn_threshold is not None:
        summary_rows.append({"metric": "knn_threshold", "value": float(query_pred.knn_threshold), "note": "kNN distance threshold"})
    if query_pred.maha_threshold is not None:
        summary_rows.append({"metric": "mahalanobis_threshold", "value": float(query_pred.maha_threshold), "note": "Mahalanobis threshold"})
    summary = pd.DataFrame(summary_rows)

    method_details = pd.DataFrame([
        {"setting": "combine_mode", "value": str(config.combine_mode)},
        {"setting": "use_williams", "value": bool(config.use_williams)},
        {"setting": "use_knn", "value": bool(config.use_knn)},
        {"setting": "use_mahalanobis", "value": bool(config.use_mahalanobis)},
        {"setting": "knn_k", "value": int(config.knn_k)},
        {"setting": "knn_quantile", "value": float(config.knn_quantile)},
        {"setting": "maha_alpha", "value": float(config.maha_alpha)},
        {"setting": "maha_use_chi2", "value": bool(config.maha_use_chi2)},
        {"setting": "features_preview", "value": ", ".join(feature_names[:30])},
    ])
    summary_dict = {
        "reference_rows": int(len(scored_ref)),
        "query_rows": int(len(scored_query)),
        "feature_count": int(len(feature_names)),
        "query_in_domain": int(scored_query["AD_in_domain"].sum()),
        "query_out_of_domain": int(len(out_domain)),
        "h_star": float(query_pred.h_star),
        "knn_threshold": None if query_pred.knn_threshold is None else float(query_pred.knn_threshold),
        "mahalanobis_threshold": None if query_pred.maha_threshold is None else float(query_pred.maha_threshold),
        "feature_names": feature_names,
        "config": {
            "combine_mode": config.combine_mode,
            "use_williams": config.use_williams,
            "use_knn": config.use_knn,
            "use_mahalanobis": config.use_mahalanobis,
        },
    }
    return ADWorkbenchResult(
        scored_query=scored_query,
        scored_reference=scored_ref,
        out_of_domain=out_domain,
        summary=summary,
        method_details=method_details,
        feature_names=feature_names,
        summary_dict=summary_dict,
    )


def write_ad_workbench_outputs(result: ADWorkbenchResult, out_prefix: str | Path) -> dict[str, str]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "query_csv": str(prefix.with_suffix(".ad_query_results.csv")),
        "reference_csv": str(prefix.with_suffix(".ad_reference_results.csv")),
        "out_of_domain_csv": str(prefix.with_suffix(".ad_out_of_domain.csv")),
        "summary_csv": str(prefix.with_suffix(".ad_summary.csv")),
        "method_csv": str(prefix.with_suffix(".ad_method_details.csv")),
        "summary_json": str(prefix.with_suffix(".ad_summary.json")),
    }
    result.scored_query.to_csv(paths["query_csv"], index=False)
    result.scored_reference.to_csv(paths["reference_csv"], index=False)
    result.out_of_domain.to_csv(paths["out_of_domain_csv"], index=False)
    result.summary.to_csv(paths["summary_csv"], index=False)
    result.method_details.to_csv(paths["method_csv"], index=False)
    Path(paths["summary_json"]).write_text(json.dumps(result.summary_dict, indent=2, sort_keys=True), encoding="utf-8")
    return paths
