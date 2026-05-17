from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False

_RDKIT_DESC_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "FractionCSP3", "HeavyAtomCount",
    "NumAromaticRings", "NumAliphaticRings", "LabuteASA",
]


def _rdkit_row(smiles: str) -> list[float]:
    if not _RDKIT_OK:
        return [np.nan] * len(_RDKIT_DESC_NAMES)
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        mol = None
    if mol is None:
        return [np.nan] * len(_RDKIT_DESC_NAMES)
    return [
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Lipinski.NumHDonors(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(mol.GetNumHeavyAtoms()),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        float(rdMolDescriptors.CalcLabuteASA(mol)),
    ]


def _find_smiles_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if col.strip().lower() in {"smiles", "smi", "canonical_smiles", "canonicalsmiles"}:
            return col
    return None


def _norm_meta_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _compute_rdkit_from_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    smiles_col = _find_smiles_col(df)
    if smiles_col is None:
        raise ValueError(
            "No numeric descriptor columns were found and no SMILES column was detected for automatic RDKit descriptor calculation. "
            "Connect Mol Descriptors 2 output to QSAR Model Hub, or include a 'SMILES' column."
        )
    rows = [_rdkit_row(s) for s in df[smiles_col].fillna("")]
    desc_df = pd.DataFrame(rows, columns=_RDKIT_DESC_NAMES, index=df.index)
    return desc_df, _RDKIT_DESC_NAMES

try:
    import optuna
    import optuna.importance
except ImportError as exc:
    optuna = None
    _OPTUNA_OK = False
    _OPTUNA_IMPORT_ERROR = str(exc)
else:
    _OPTUNA_OK = True
    _OPTUNA_IMPORT_ERROR = ""
    optuna.logging.set_verbosity(logging.WARNING)

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from chem_inf_widgets.chemcore.services.safe_feature_selection import safe_f_regression

try:
    import lightgbm as _lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False

try:
    import xgboost as _xgb
    _XGB_OK = True
except ImportError:
    _XGB_OK = False


def _make_registry() -> dict:
    reg = {
        "random_forest":    RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1),
        "extra_trees":      ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "ridge":            Ridge(alpha=1.0),
        "elastic_net":      ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=42, max_iter=10000),
        "linear":           LinearRegression(),
        "pls":              PLSRegression(n_components=2),
        "svr_rbf":          SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1),
    }
    if _LGB_OK:
        reg["lightgbm"] = _lgb.LGBMRegressor(n_estimators=300, random_state=42, verbose=-1, n_jobs=1)
    if _XGB_OK:
        reg["xgboost"] = _xgb.XGBRegressor(n_estimators=300, random_state=42, verbosity=0, n_jobs=1)
    return reg

MODEL_REGISTRY = _make_registry()

SCALE_DEFAULT = {"ridge", "elastic_net", "linear", "pls", "svr_rbf"}

# Algorithms included in HPO search
_HPO_ALGORITHMS: list[str] = [
    "random_forest", "extra_trees", "gradient_boosting",
    "ridge", "elastic_net", "svr_rbf",
] + (["lightgbm"] if _LGB_OK else []) + (["xgboost"] if _XGB_OK else [])

_SAMPLERS: dict[str, Any] = (
    {
        "tpe":    lambda seed: optuna.samplers.TPESampler(seed=seed),
        "cmaes":  lambda seed: optuna.samplers.CmaEsSampler(seed=seed),
        "gp":     lambda seed: optuna.samplers.GPSampler(seed=seed),
        "qmc":    lambda seed: optuna.samplers.QMCSampler(seed=seed),
        "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
    }
    if _OPTUNA_OK else {}
)

_PRUNERS: dict[str, Any] = (
    {
        "median": lambda: optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        "none":   lambda: optuna.pruners.NopPruner(),
    }
    if _OPTUNA_OK else {}
)


@dataclass(frozen=True)
class QSARModelHubConfig:
    target_column: str = "pActivity"
    id_column: str = "compound_id"
    model_key: str = "random_forest"
    test_size: float = 0.25
    cv_folds: int = 5
    random_state: int = 42
    scale_features: Optional[bool] = None
    drop_constant_features: bool = True
    min_non_missing_fraction: float = 0.70
    # HPO (Optuna)
    use_hpo: bool = False
    hpo_trials: int = 50
    hpo_timeout: Optional[float] = None
    hpo_sampler: str = "tpe"          # tpe | cmaes | gp | qmc | random
    hpo_pruner: str = "median"        # median | none
    use_feature_selection: bool = False
    fs_max_features: int = 50
    ensemble_top_k: int = 0           # 0 = disabled; >0 = average top-k trial models


@dataclass(frozen=True)
class QSARModelHubResult:
    model_key: str
    target_column: str
    id_column: str
    feature_names: list[str]
    n_rows_input: int
    n_rows_used: int
    n_features_input: int
    n_features_used: int
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    cv_metrics: dict[str, float]
    predictions: pd.DataFrame
    metrics_table: pd.DataFrame
    summary: dict[str, Any]
    pipeline: Any
    hpo_history: Optional[pd.DataFrame] = None
    best_params: Optional[dict[str, Any]] = None
    param_importances: Optional[dict[str, float]] = None


def hpo_available() -> bool:
    return _OPTUNA_OK


def _require_optuna() -> None:
    if _OPTUNA_OK:
        return
    raise RuntimeError(
        "Optuna is not installed. Install 'optuna' to use Hyperparameter Optimization "
        "or the 'auto' model in QSAR Model Hub."
    )


def available_model_keys() -> list[str]:
    keys = list(MODEL_REGISTRY.keys())
    return (["auto"] + keys) if _OPTUNA_OK else keys


class _EnsemblePipeline:
    """Average predictions from N independently-fitted pipelines."""
    def __init__(self, pipelines: list) -> None:
        self.pipelines = pipelines
        # mirror sklearn API
        self.named_steps = pipelines[0].named_steps if pipelines else {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([p.predict(X) for p in self.pipelines])
        return np.mean(preds, axis=0)


def _suggest_model(trial: "optuna.Trial", algorithms: list[str] | None = None) -> tuple[str, Any]:
    """Return (algorithm_key, unfitted_estimator) with Optuna-suggested hyperparameters."""
    algos = algorithms or _HPO_ALGORITHMS
    algo = trial.suggest_categorical("algorithm", algos)

    if algo == "random_forest":
        est = RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("rf_max_depth", 3, 30),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("rf_max_features", 0.1, 1.0),
            random_state=42, n_jobs=1,
        )
    elif algo == "extra_trees":
        est = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("et_n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("et_max_depth", 3, 30),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 10),
            random_state=42, n_jobs=1,
        )
    elif algo == "gradient_boosting":
        est = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("gb_n_estimators", 50, 400, step=50),
            learning_rate=trial.suggest_float("gb_learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("gb_max_depth", 2, 8),
            subsample=trial.suggest_float("gb_subsample", 0.5, 1.0),
            random_state=42,
        )
    elif algo == "ridge":
        est = Ridge(alpha=trial.suggest_float("ridge_alpha", 1e-4, 1e4, log=True))
    elif algo == "elastic_net":
        est = ElasticNet(
            alpha=trial.suggest_float("en_alpha", 1e-5, 10.0, log=True),
            l1_ratio=trial.suggest_float("en_l1_ratio", 0.0, 1.0),
            max_iter=10000, random_state=42,
        )
    elif algo == "svr_rbf":
        est = SVR(
            C=trial.suggest_float("svr_C", 1e-2, 1e3, log=True),
            gamma=trial.suggest_categorical("svr_gamma", ["scale", "auto"]),
            epsilon=trial.suggest_float("svr_epsilon", 1e-3, 1.0, log=True),
        )
    elif algo == "lightgbm" and _LGB_OK:
        est = _lgb.LGBMRegressor(
            n_estimators=trial.suggest_int("lgb_n_estimators", 50, 500, step=50),
            learning_rate=trial.suggest_float("lgb_learning_rate", 1e-3, 0.3, log=True),
            num_leaves=trial.suggest_int("lgb_num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("lgb_min_child_samples", 5, 50),
            subsample=trial.suggest_float("lgb_subsample", 0.5, 1.0),
            random_state=42, verbose=-1, n_jobs=1,
        )
    elif algo == "xgboost" and _XGB_OK:
        est = _xgb.XGBRegressor(
            n_estimators=trial.suggest_int("xgb_n_estimators", 50, 500, step=50),
            learning_rate=trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 10),
            subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.4, 1.0),
            random_state=42, verbosity=0, n_jobs=1,
        )
    else:
        est = LinearRegression()
    return algo, est


def _build_trial_pipeline(algo_key: str, estimator: Any, scale: bool,
                           use_fs: bool, n_features: int, k: int) -> Pipeline:
    steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    if use_fs and n_features > k:
        steps.append(("selector", SelectKBest(safe_f_regression, k=k)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _run_hpo(
    X: np.ndarray,
    y: np.ndarray,
    config: QSARModelHubConfig,
) -> tuple[Any, str, dict[str, Any], pd.DataFrame, Optional[dict[str, float]]]:
    """
    Run Optuna HPO.
    Returns (best_pipeline_or_ensemble, best_algo_key, best_params, history_df, param_importances).
    """
    _require_optuna()
    n_features = X.shape[1]
    cv = KFold(n_splits=min(int(config.cv_folds), len(y)), shuffle=True, random_state=int(config.random_state))
    use_fs = bool(config.use_feature_selection)
    k_fs = max(1, min(int(config.fs_max_features), n_features))

    def objective(trial: "optuna.Trial") -> float:
        algo_key, estimator = _suggest_model(trial)
        scale = algo_key in SCALE_DEFAULT
        if use_fs:
            k = trial.suggest_int("n_features_selected", max(1, k_fs // 2), k_fs)
        else:
            k = k_fs
        pipe = _build_trial_pipeline(algo_key, estimator, scale, use_fs, n_features, k)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=1, error_score="raise")
        return float(np.mean(scores))

    sampler_fn = _SAMPLERS.get(config.hpo_sampler, _SAMPLERS["tpe"])
    pruner_fn  = _PRUNERS.get(config.hpo_pruner,   _PRUNERS["median"])
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler_fn(int(config.random_state)),
        pruner=pruner_fn(),
    )
    study.optimize(
        objective,
        n_trials=int(config.hpo_trials),
        timeout=config.hpo_timeout,
        show_progress_bar=False,
    )

    # Best trial → pipeline
    best = study.best_trial
    best_algo, best_est = _suggest_model(optuna.trial.FixedTrial(best.params))
    scale = best_algo in SCALE_DEFAULT
    k_best = best.params.get("n_features_selected", k_fs) if use_fs else k_fs
    best_pipeline = _build_trial_pipeline(best_algo, best_est, scale, use_fs, n_features, k_best)

    # Ensemble: take top-k trial pipelines (each fitted on full X)
    ensemble_k = int(config.ensemble_top_k)
    if ensemble_k > 1:
        sorted_trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value, reverse=True,
        )[:ensemble_k]
        ens_pipes = []
        for t in sorted_trials:
            a, e = _suggest_model(optuna.trial.FixedTrial(t.params))
            s = a in SCALE_DEFAULT
            k_t = t.params.get("n_features_selected", k_fs) if use_fs else k_fs
            p = _build_trial_pipeline(a, e, s, use_fs, n_features, k_t)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                p.fit(X, y)
            ens_pipes.append(p)
        final_pipeline: Any = _EnsemblePipeline(ens_pipes)
    else:
        final_pipeline = best_pipeline

    # History dataframe
    history_rows = []
    for t in study.trials:
        row = {"trial": t.number, "value_cv_r2": t.value if t.value is not None else float("nan")}
        row.update(t.params)
        history_rows.append(row)
    history_df = pd.DataFrame(history_rows)

    # Hyperparameter importances (fANOVA)
    try:
        param_imp: Optional[dict[str, float]] = {
            k: float(v) for k, v in optuna.importance.get_param_importances(study).items()
        }
    except Exception:
        param_imp = None

    return final_pipeline, best_algo, dict(best.params), history_df, param_imp


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {f"{prefix}r2": np.nan, f"{prefix}rmse": np.nan, f"{prefix}mae": np.nan, f"{prefix}n": 0}
    yt = y_true[mask]
    yp = y_pred[mask]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else np.nan
    return {f"{prefix}r2": r2, f"{prefix}rmse": rmse, f"{prefix}mae": mae, f"{prefix}n": int(len(yt))}


def _select_feature_columns(df: pd.DataFrame, *, target_column: str, id_column: str) -> list[str]:
    blocked = {target_column, id_column}
    cols: list[str] = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _is_system_or_provenance_feature(col: str) -> bool:
    normalized = _norm_meta_name(col)
    if normalized.startswith("source_"):
        return True
    blocked_exact = {
        "row_id",
        "transform_log",
        "qc_flags",
        "dropped_reason",
        "source_row_ids",
        "source_transform_logs",
        "source_qc_flags_all",
        "source_dropped_reasons",
        "source_row_id",
        "source_transform_log",
        "source_qc_flags",
        "source_dropped_reason",
        "import_source_format",
        "source_format",
        "source_name",
    }
    return normalized in blocked_exact


def _filter_features(df: pd.DataFrame, feature_cols: list[str], config: QSARModelHubConfig) -> list[str]:
    keep: list[str] = []
    min_frac = float(config.min_non_missing_fraction)
    for col in feature_cols:
        if _is_system_or_provenance_feature(col):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().mean() < min_frac:
            continue
        if config.drop_constant_features and s.nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    return keep


def _prediction_passthrough_columns(
    df: pd.DataFrame,
    *,
    target_column: str,
    id_column: str,
) -> list[str]:
    wanted = {
        "smiles",
        "canonical_smiles",
        "compound_name",
        "name",
        "row_id",
        "chembl_id",
        "molecule_chembl_id",
    }
    excluded = {_norm_meta_name(target_column), _norm_meta_name(id_column)}
    cols: list[str] = []
    for col in df.columns:
        normalized = _norm_meta_name(col)
        if normalized in excluded:
            continue
        if normalized in wanted:
            cols.append(col)
    return cols


def _build_pipeline(model_key: str, scale_features: bool) -> Pipeline:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key '{model_key}'. Available: {', '.join(available_model_keys())}")
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(MODEL_REGISTRY[model_key])))
    return Pipeline(steps)


def train_qsar_model_hub(df: pd.DataFrame, config: QSARModelHubConfig) -> QSARModelHubResult:
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found. Available columns: {', '.join(map(str, df.columns))}")
    data = df.copy()
    data[config.target_column] = pd.to_numeric(data[config.target_column], errors="coerce")
    feature_input = _select_feature_columns(data, target_column=config.target_column, id_column=config.id_column)
    auto_descriptors = False
    if not feature_input:
        # No numeric features yet — try to auto-compute 12 compact RDKit descriptors from SMILES.
        desc_df, rdkit_names = _compute_rdkit_from_df(data)
        for col in rdkit_names:
            data[col] = desc_df[col]
        feature_input = rdkit_names
        auto_descriptors = True
    feature_cols = _filter_features(data, feature_input, config)
    if not feature_cols:
        raise ValueError("No usable feature columns remained after missing-value/constant-feature filtering.")
    usable = data[data[config.target_column].notna()].copy()
    if len(usable) < 4:
        raise ValueError("At least 4 rows with numeric target values are required.")
    X = usable[feature_cols].to_numpy(dtype=float)
    y = usable[config.target_column].to_numpy(dtype=float)
    ids = usable[config.id_column].astype(str).to_numpy() if config.id_column in usable.columns else np.arange(len(usable)).astype(str)
    row_positions = np.arange(len(usable), dtype=int)
    passthrough_cols = _prediction_passthrough_columns(
        usable,
        target_column=config.target_column,
        id_column=config.id_column,
    )

    hpo_history: Optional[pd.DataFrame] = None
    best_params: Optional[dict[str, Any]] = None
    effective_key = config.model_key

    param_importances: Optional[dict[str, float]] = None
    if config.use_hpo or config.model_key == "auto":
        _require_optuna()
        pipeline, effective_key, best_params, hpo_history, param_importances = _run_hpo(X, y, config)
        scale = effective_key in SCALE_DEFAULT
    else:
        scale = (config.model_key in SCALE_DEFAULT) if config.scale_features is None else bool(config.scale_features)
        pipeline = _build_pipeline(config.model_key, scale)

    test_size = min(max(float(config.test_size), 0.05), 0.8)
    if len(usable) < 8:
        # tiny teaching/demo sets are more useful with a deterministic small split
        test_size = min(max(1 / len(usable), test_size), 0.5)
    X_train, X_test, y_train, y_test, ids_train, ids_test, pos_train, pos_test = train_test_split(
        X,
        y,
        ids,
        row_positions,
        test_size=test_size,
        random_state=int(config.random_state),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipeline.fit(X_train, y_train)
    pred_train = pipeline.predict(X_train).ravel()
    pred_test = pipeline.predict(X_test).ravel()

    cv_metrics: dict[str, float]
    if len(usable) >= max(3, int(config.cv_folds)):
        n_splits = min(int(config.cv_folds), len(usable))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(config.random_state))
        cv_pipe = pipeline if (config.use_hpo or config.model_key == "auto") else _build_pipeline(effective_key, scale)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cv_pred = cross_val_predict(cv_pipe, X, y, cv=cv).ravel()
        cv_metrics = _regression_metrics(y, cv_pred, prefix="cv_")
    else:
        cv_pred = np.full_like(y, np.nan, dtype=float)
        cv_metrics = {"cv_r2": np.nan, "cv_rmse": np.nan, "cv_mae": np.nan, "cv_n": 0}

    train_metrics = _regression_metrics(y_train, pred_train, prefix="train_")
    test_metrics = _regression_metrics(y_test, pred_test, prefix="test_")

    pred_rows = []
    for split_name, split_ids, actual, pred, positions in (
        ("train", ids_train, y_train, pred_train, pos_train),
        ("test", ids_test, y_test, pred_test, pos_test),
    ):
        split_meta = usable.iloc[np.asarray(positions, dtype=int)]
        meta_records = split_meta[passthrough_cols].to_dict("records") if passthrough_cols else [{} for _ in range(len(split_meta))]
        for cid, a, p, meta in zip(split_ids, actual, pred, meta_records):
            row = {
                "compound_id": cid,
                "split": split_name,
                "observed": float(a),
                "predicted": float(p),
                "residual": float(a - p),
                "abs_residual": float(abs(a - p)),
                "model": effective_key,
            }
            for col in passthrough_cols:
                value = meta.get(col, "")
                row[col] = "" if pd.isna(value) else value
            pred_rows.append(row)
    predictions = pd.DataFrame(pred_rows)

    metric_rows = []
    for group, metrics in (("train", train_metrics), ("test", test_metrics), ("cross_validation", cv_metrics)):
        for metric, value in metrics.items():
            metric_rows.append({"group": group, "metric": metric, "value": value})
    metrics_table = pd.DataFrame(metric_rows)

    summary = {
        "model_key": effective_key,
        "hpo_used": bool(config.use_hpo or config.model_key == "auto"),
        "hpo_trials": int(config.hpo_trials) if (config.use_hpo or config.model_key == "auto") else 0,
        "best_params": best_params or {},
        "target_column": config.target_column,
        "id_column": config.id_column,
        "n_rows_input": int(len(df)),
        "n_rows_used": int(len(usable)),
        "n_features_input": int(len(feature_input)),
        "n_features_used": int(len(feature_cols)),
        "scale_features": bool(scale),
        "drop_constant_features": bool(config.drop_constant_features),
        "min_non_missing_fraction": float(config.min_non_missing_fraction),
        "auto_rdkit_descriptors": bool(auto_descriptors),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "feature_names_preview": feature_cols[:25],
    }
    return QSARModelHubResult(
        model_key=effective_key,
        target_column=config.target_column,
        id_column=config.id_column,
        feature_names=feature_cols,
        n_rows_input=int(len(df)),
        n_rows_used=int(len(usable)),
        n_features_input=int(len(feature_input)),
        n_features_used=int(len(feature_cols)),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        cv_metrics=cv_metrics,
        predictions=predictions,
        metrics_table=metrics_table,
        summary=summary,
        pipeline=pipeline,
        hpo_history=hpo_history,
        best_params=best_params,
        param_importances=param_importances,
    )


def write_qsar_model_hub_outputs(result: QSARModelHubResult, out_prefix: str | Path) -> dict[str, str]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions_csv": str(prefix.with_suffix(".predictions.csv")),
        "metrics_csv": str(prefix.with_suffix(".metrics.csv")),
        "summary_json": str(prefix.with_suffix(".summary.json")),
        "features_txt": str(prefix.with_suffix(".features.txt")),
    }
    result.predictions.to_csv(paths["predictions_csv"], index=False)
    result.metrics_table.to_csv(paths["metrics_csv"], index=False)
    Path(paths["summary_json"]).write_text(json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8")
    Path(paths["features_txt"]).write_text("\n".join(result.feature_names) + "\n", encoding="utf-8")
    return paths
