from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


DEFAULT_BLOCKED = {
    "compound_id", "molecule_id", "id", "name", "smiles", "canonical_smiles",
    "standardized_smiles", "inchi", "inchikey", "standard_inchikey", "split",
}


@dataclass(frozen=True)
class ModelExplanationConfig:
    target_column: str = "pActivity"
    id_column: str = "compound_id"
    max_features: int = 50
    method: str = "auto"  # auto, model_importance, coefficient, permutation, univariate
    n_repeats: int = 8
    random_state: int = 42
    min_non_missing_fraction: float = 0.70
    drop_constant_features: bool = True


@dataclass(frozen=True)
class ModelExplanationResult:
    feature_importance: pd.DataFrame
    local_contributions: pd.DataFrame
    feature_summary: pd.DataFrame
    explanation_summary: pd.DataFrame
    summary_dict: dict[str, Any]


def _select_features(df: pd.DataFrame, cfg: ModelExplanationConfig) -> list[str]:
    blocked = {cfg.target_column, cfg.id_column} | DEFAULT_BLOCKED
    features: list[str] = []
    for col in df.columns:
        if col in blocked or str(col).lower() in blocked:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() < float(cfg.min_non_missing_fraction):
            continue
        if cfg.drop_constant_features and s.nunique(dropna=True) <= 1:
            continue
        if s.notna().sum() > 0:
            features.append(str(col))
    return features


def _model_step(model: Any) -> Any:
    if model is None:
        return None
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def _fit_fallback_model(X: np.ndarray, y: np.ndarray, random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=300, random_state=int(random_state), n_jobs=-1)),
    ]).fit(X, y)


def _coerce_model(model: Any, X: np.ndarray, y: np.ndarray, cfg: ModelExplanationConfig) -> Any:
    if model is not None:
        return model
    return _fit_fallback_model(X, y, cfg.random_state)


def _importance_from_model(model: Any, feature_names: list[str]) -> tuple[np.ndarray | None, str]:
    step = _model_step(model)
    if step is None:
        return None, "none"
    if hasattr(step, "feature_importances_"):
        return np.asarray(step.feature_importances_, dtype=float).ravel(), "model_importance"
    if hasattr(step, "coef_"):
        coef = np.asarray(step.coef_, dtype=float).ravel()
        return np.abs(coef), "coefficient_abs"
    return None, "unsupported"


def _univariate_importance(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> np.ndarray:
    vals = []
    for j in range(X.shape[1]):
        x = X[:, j]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or np.nanstd(x[mask]) <= 1e-12:
            vals.append(0.0)
        else:
            vals.append(abs(float(np.corrcoef(x[mask], y[mask])[0, 1])))
    return np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0)


def _normalized(vals: np.ndarray) -> np.ndarray:
    vals = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    s = float(np.sum(np.abs(vals)))
    return vals / s if s > 0 else vals


def explain_qsar_model(
    df: pd.DataFrame,
    model: Any | None = None,
    config: ModelExplanationConfig = ModelExplanationConfig(),
) -> ModelExplanationResult:
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found.")
    data = df.copy()
    data[config.target_column] = pd.to_numeric(data[config.target_column], errors="coerce")
    feature_names = _select_features(data, config)
    if not feature_names:
        raise ValueError("No numeric descriptor/fingerprint feature columns were detected.")
    usable = data[data[config.target_column].notna()].copy()
    if len(usable) < 4:
        raise ValueError("At least 4 rows with a numeric target are required for model explanation.")
    X_df = usable[feature_names].apply(pd.to_numeric, errors="coerce")
    X = X_df.to_numpy(dtype=float)
    y = usable[config.target_column].to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    fitted_model = _coerce_model(model, X, y, config)
    requested = (config.method or "auto").strip().lower()
    importance = None
    method_used = ""

    if requested in {"auto", "model_importance", "coefficient"}:
        importance, method_used = _importance_from_model(fitted_model, feature_names)
        if importance is not None and len(importance) != len(feature_names):
            importance = None
            method_used = "unsupported_feature_count"
        if requested != "auto" and importance is None:
            raise ValueError(f"The supplied model does not expose compatible importances for method '{requested}'.")

    if importance is None and requested in {"auto", "permutation"}:
        try:
            perm = permutation_importance(
                fitted_model, X, y, n_repeats=int(config.n_repeats), random_state=int(config.random_state), n_jobs=1
            )
            importance = np.maximum(perm.importances_mean, 0.0)
            method_used = "permutation"
        except Exception:
            if requested == "permutation":
                raise

    if importance is None:
        importance = _univariate_importance(X_imp, y, feature_names)
        method_used = "univariate_correlation"

    norm = _normalized(importance)
    order = np.argsort(-np.abs(norm))
    max_features = max(1, int(config.max_features))
    rows = []
    for rank, idx in enumerate(order[:max_features], start=1):
        feature = feature_names[int(idx)]
        series = X_df[feature]
        rows.append({
            "rank": rank,
            "feature": feature,
            "importance": float(importance[int(idx)]),
            "normalized_importance": float(norm[int(idx)]),
            "method": method_used,
            "mean": float(series.mean(skipna=True)),
            "std": float(series.std(skipna=True)),
            "missing_fraction": float(series.isna().mean()),
        })
    feature_importance = pd.DataFrame(rows)

    # Simple local explanation: centered feature value * global normalized importance.
    top_features = [r["feature"] for r in rows[: min(20, len(rows))]]
    centers = X_df[top_features].mean(skipna=True)
    scales = X_df[top_features].std(skipna=True).replace(0, 1.0).fillna(1.0)
    weight_map = dict(zip(feature_importance["feature"], feature_importance["normalized_importance"]))
    local_rows = []
    ids = usable[config.id_column].astype(str).tolist() if config.id_column in usable.columns else [str(i) for i in range(len(usable))]
    for row_i, (_idx, row) in enumerate(usable.iterrows()):
        contrib = {}
        for feat in top_features:
            val = pd.to_numeric(pd.Series([row.get(feat)]), errors="coerce").iloc[0]
            z = 0.0 if not np.isfinite(val) else float((val - centers[feat]) / scales[feat])
            contrib[feat] = z * float(weight_map.get(feat, 0.0))
        if contrib:
            best = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            local_rows.append({
                "compound_id": ids[row_i],
                "top_contributing_features": "; ".join(f"{k}:{v:.4g}" for k, v in best),
                "approx_local_score": float(sum(contrib.values())),
            })
    local_contributions = pd.DataFrame(local_rows)

    feature_summary = pd.DataFrame({
        "metric": ["rows_used", "features_used", "features_reported", "method_used"],
        "value": [int(len(usable)), int(len(feature_names)), int(len(feature_importance)), method_used],
    })
    explanation_summary = pd.DataFrame([
        {"item": "method", "value": method_used},
        {"item": "target_column", "value": config.target_column},
        {"item": "id_column", "value": config.id_column},
        {"item": "n_rows_used", "value": int(len(usable))},
        {"item": "n_features_used", "value": int(len(feature_names))},
        {"item": "top_feature", "value": feature_importance.iloc[0]["feature"] if len(feature_importance) else ""},
    ])
    summary_dict = {
        "method_used": method_used,
        "target_column": config.target_column,
        "id_column": config.id_column,
        "n_rows_used": int(len(usable)),
        "n_features_used": int(len(feature_names)),
        "features_reported": int(len(feature_importance)),
        "top_features": feature_importance[["feature", "importance", "normalized_importance"]].head(20).to_dict(orient="records"),
    }
    return ModelExplanationResult(
        feature_importance=feature_importance,
        local_contributions=local_contributions,
        feature_summary=feature_summary,
        explanation_summary=explanation_summary,
        summary_dict=summary_dict,
    )


def write_model_explanation_outputs(result: ModelExplanationResult, out_prefix: str | Path) -> dict[str, str]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_importance_csv": str(prefix.with_suffix(".feature_importance.csv")),
        "local_contributions_csv": str(prefix.with_suffix(".local_contributions.csv")),
        "feature_summary_csv": str(prefix.with_suffix(".feature_summary.csv")),
        "explanation_summary_csv": str(prefix.with_suffix(".explanation_summary.csv")),
        "summary_json": str(prefix.with_suffix(".model_explanation_summary.json")),
    }
    result.feature_importance.to_csv(paths["feature_importance_csv"], index=False)
    result.local_contributions.to_csv(paths["local_contributions_csv"], index=False)
    result.feature_summary.to_csv(paths["feature_summary_csv"], index=False)
    result.explanation_summary.to_csv(paths["explanation_summary_csv"], index=False)
    Path(paths["summary_json"]).write_text(json.dumps(result.summary_dict, indent=2, sort_keys=True), encoding="utf-8")
    return paths
