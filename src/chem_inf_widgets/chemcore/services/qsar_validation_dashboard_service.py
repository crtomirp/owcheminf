from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class QSARValidationConfig:
    observed_column: str = "observed"
    predicted_column: str = "predicted"
    split_column: str = "split"
    id_column: str = "compound_id"
    residual_threshold: float | None = None
    z_threshold: float = 3.0


@dataclass(frozen=True)
class QSARValidationResult:
    metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    outliers: pd.DataFrame
    summary: dict[str, Any]


def _metrics_for_group(group_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {"group": group_name, "n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan, "bias": np.nan}
    residual = yt - yp
    return {
        "group": group_name,
        "n": int(len(yt)),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "bias": float(np.mean(residual)),
        "residual_sd": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def validate_qsar_predictions(df: pd.DataFrame, config: QSARValidationConfig = QSARValidationConfig()) -> QSARValidationResult:
    for col in (config.observed_column, config.predicted_column):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available columns: {', '.join(map(str, df.columns))}")
    data = df.copy()
    data[config.observed_column] = pd.to_numeric(data[config.observed_column], errors="coerce")
    data[config.predicted_column] = pd.to_numeric(data[config.predicted_column], errors="coerce")
    data["residual"] = data[config.observed_column] - data[config.predicted_column]
    data["abs_residual"] = data["residual"].abs()
    residual_sd = float(data["residual"].std(ddof=1)) if data["residual"].notna().sum() > 1 else 0.0
    residual_mean = float(data["residual"].mean()) if data["residual"].notna().sum() else 0.0
    if residual_sd > 0:
        data["residual_z"] = (data["residual"] - residual_mean) / residual_sd
    else:
        data["residual_z"] = 0.0
    if config.residual_threshold is None:
        threshold = float(max(1.0, 2.0 * residual_sd)) if residual_sd > 0 else 1.0
    else:
        threshold = float(config.residual_threshold)
    data["large_residual"] = data["abs_residual"] > threshold
    data["z_outlier"] = data["residual_z"].abs() > float(config.z_threshold)
    data["validation_flag"] = np.where(data["large_residual"] | data["z_outlier"], "review", "ok")

    metric_rows = [_metrics_for_group("all", data[config.observed_column].to_numpy(float), data[config.predicted_column].to_numpy(float))]
    if config.split_column in data.columns:
        for split, sub in data.groupby(config.split_column):
            metric_rows.append(_metrics_for_group(str(split), sub[config.observed_column].to_numpy(float), sub[config.predicted_column].to_numpy(float)))
    metrics = pd.DataFrame(metric_rows)
    outliers = data[data["validation_flag"] == "review"].copy()
    summary = {
        "n_rows": int(len(data)),
        "n_outliers": int(len(outliers)),
        "residual_threshold": threshold,
        "z_threshold": float(config.z_threshold),
        "overall_metrics": metric_rows[0],
    }
    return QSARValidationResult(metrics=metrics, diagnostics=data, outliers=outliers, summary=summary)


def write_qsar_validation_outputs(result: QSARValidationResult, out_prefix: str | Path) -> dict[str, str]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_csv": str(prefix.with_suffix(".validation_metrics.csv")),
        "diagnostics_csv": str(prefix.with_suffix(".residual_diagnostics.csv")),
        "outliers_csv": str(prefix.with_suffix(".outliers.csv")),
        "summary_json": str(prefix.with_suffix(".validation_summary.json")),
    }
    result.metrics.to_csv(paths["metrics_csv"], index=False)
    result.diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    result.outliers.to_csv(paths["outliers_csv"], index=False)
    Path(paths["summary_json"]).write_text(json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8")
    return paths
