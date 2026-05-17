from __future__ import annotations

import io
import json
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.path import Path
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning

from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table
from rdkit.Chem import Draw
from chem_inf_widgets.chemcore.services.safe_feature_selection import safe_f_regression
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
except Exception:  # pragma: no cover - optional RDKit guard
    Chem = None
    Descriptors = None
    Crippen = None
    Lipinski = None
    rdMolDescriptors = None

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles
from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table, safe_table_from_numpy

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_size=256, epochs=200, lr=0.01, batch_size=32, random_state=42):
        self.hidden_layer_size = hidden_layer_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("Deep Learning Regression requires the optional 'torch' package.")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        input_dim = X_tensor.shape[1]

        self.model_ = nn.Sequential(
            nn.Linear(input_dim, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, 1),
        )
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        if not TORCH_AVAILABLE:
            raise ImportError("Deep Learning Regression requires the optional 'torch' package.")
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model_(X_tensor)
        return predictions.numpy().ravel()


@dataclass(frozen=True)
class QSARRunConfig:
    selected_algorithm: int
    normalization_method: int
    imputation_method: int
    cv_folds: int
    test_size: float
    tuning_method: int
    n_iter: int
    hyperparameters: str
    enable_feature_selection: bool
    num_features: int
    max_model_features: int
    enable_applicability_domain: bool
    enable_auto_qsar: bool
    algorithms: Sequence[tuple[str, type]]


@dataclass(frozen=True)
class CompoundPreview:
    title: str
    png_bytes: bytes


@dataclass(frozen=True)
class DiagnosticPlotData:
    preds: np.ndarray
    actuals: np.ndarray
    residuals: np.ndarray
    inlier_mask: np.ndarray
    outlier_mask: np.ndarray
    is_classification: bool


@dataclass(frozen=True)
class DiagnosticSeries:
    x: np.ndarray
    y: np.ndarray
    color: str
    label: str


@dataclass(frozen=True)
class DiagnosticPlotSpec:
    left_series: tuple[DiagnosticSeries, ...]
    right_series: tuple[DiagnosticSeries, ...]
    diagonal_min: float
    diagonal_max: float
    left_title: str
    left_xlabel: str
    left_ylabel: str
    right_title: str
    right_xlabel: str
    right_ylabel: str
    show_legends: bool


@dataclass(frozen=True)
class DiagnosticDatasetPayload:
    dataset_type: str
    X: np.ndarray
    y: np.ndarray
    pipeline: object
    is_classification: bool
    result_table: Optional[Table]


@dataclass(frozen=True)
class ReportContext:
    model_name: str
    total_descriptors: int
    descriptors_used: int
    cv_score: Optional[float]
    train_metrics: dict
    test_metrics: dict
    external_metrics: dict


@dataclass(frozen=True)
class FeatureInspectionPayload:
    available: bool
    message_html: str
    value_label: str
    names: tuple[str, ...]
    values: Optional[np.ndarray]
    ses: Optional[np.ndarray]
    ts: Optional[np.ndarray]
    ps: Optional[np.ndarray]
    vifs: Optional[np.ndarray]
    chart_names: tuple[str, ...]
    chart_values: Optional[np.ndarray]
    chart_colors: tuple[str, ...]
    chart_title: str
    subtitle: str
    tab_title: str


@dataclass(frozen=True)
class SelectionGalleryPayload:
    placeholder_text: Optional[str]
    previews: tuple[CompoundPreview, ...]
    more_count: int


@dataclass(frozen=True)
class SelectionPublishPayload:
    selected_table: Table
    gallery: SelectionGalleryPayload
    status_text: str


def available_algorithms():
    algorithms = [
        ("Random Forest", RandomForestRegressor),
        ("Support Vector Regression", SVR),
        ("Gradient Boosting", GradientBoostingRegressor),
        ("PLS Regression", PLSRegression),
        ("Decision Tree Regression", DecisionTreeRegressor),
        ("Lasso Regression", Lasso),
        ("Ridge Regression", Ridge),
        ("Elastic Net", ElasticNet),
    ]
    if TORCH_AVAILABLE:
        algorithms.append(("Deep Learning Regression", TorchRegressor))
    return algorithms


def build_run_config(
    *,
    selected_algorithm: int,
    normalization_method: int,
    imputation_method: int,
    cv_folds: int,
    test_size: float,
    tuning_method: int,
    n_iter: int,
    hyperparameters: str,
    enable_feature_selection: bool,
    num_features: int,
    algorithms: Sequence[tuple[str, type]],
    max_model_features: int = 1000,
    enable_applicability_domain: bool = True,
    enable_auto_qsar: bool = False,
) -> QSARRunConfig:
    return QSARRunConfig(
        selected_algorithm=selected_algorithm,
        normalization_method=normalization_method,
        imputation_method=imputation_method,
        cv_folds=cv_folds,
        test_size=test_size,
        tuning_method=tuning_method,
        n_iter=n_iter,
        hyperparameters=hyperparameters,
        enable_feature_selection=enable_feature_selection,
        num_features=num_features,
        max_model_features=int(max_model_features or 0),
        enable_applicability_domain=bool(enable_applicability_domain),
        enable_auto_qsar=bool(enable_auto_qsar),
        algorithms=algorithms,
    )


def find_smiles_var(data: Table):
    wanted = {"smiles", "canonical_smiles", "smile"}
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    preferred = [var for var in variables if isinstance(var, StringVariable) and var.name.strip().lower() in wanted]
    if preferred:
        return preferred[0]
    return next((var for var in variables if isinstance(var, StringVariable)), None)


def find_name_var(data: Table):
    wanted = {"name", "title", "compound", "compound_name"}
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    return next(
        (var for var in variables if isinstance(var, StringVariable) and var.name.strip().lower() in wanted),
        None,
    )

TARGET_COLUMN_CANDIDATES = {
    "pactivity",
    "p_activity",
    "pchembl_value",
    "pic50",
    "pki",
    "pkd",
    "pec50",
    "activity",
}

LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES = {
    "row_index",
    "activity",
    "activity_value",
    "pactivity",
    "p_activity",
    "pactivity_raw",
    "pactivity_min",
    "pactivity_max",
    "pactivity_std",
    "pchembl_value",
    "pic50",
    "pki",
    "pkd",
    "pec50",
    "n_measurements",
    "duplicate_group",
}

RDKit_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "LabuteASA",
]


def _norm_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _is_numeric_variable(var) -> bool:
    return isinstance(var, ContinuousVariable) or isinstance(var, DiscreteVariable)


def _target_candidate_var(data: Table):
    variables = list(data.domain.class_vars) + list(data.domain.attributes) + list(data.domain.metas)
    for var in variables:
        if _norm_name(var.name) in TARGET_COLUMN_CANDIDATES and _is_numeric_variable(var):
            return var
    # String meta fallback, useful for imported CSV values that were not typed correctly.
    for var in variables:
        if _norm_name(var.name) in TARGET_COLUMN_CANDIDATES:
            try:
                col = data.get_column(var)
                vals = np.asarray([float(str(v).strip().replace(",", ".")) for v in col if str(v).strip()], dtype=float)
                if vals.size:
                    return var
            except Exception:
                pass
    return None


def _column_as_float(data: Table, var) -> np.ndarray:
    col = data.get_column(var)
    out = []
    for value in col:
        if value is None:
            out.append(np.nan)
            continue
        try:
            out.append(float(value))
        except Exception:
            try:
                out.append(float(str(value).strip().replace(",", ".")))
            except Exception:
                out.append(np.nan)
    return np.asarray(out, dtype=float)


def _numeric_columns_from_vars(data: Table, variables: Sequence) -> tuple[np.ndarray, list[str]]:
    """Convert variables to a numeric matrix, accepting numeric strings from metas.

    This is intentionally conservative: a column is kept only if at least one
    value is numeric. Fully non-numeric columns such as SMILES/name are ignored.
    """
    cols = []
    names = []
    for var in variables:
        arr = _column_as_float(data, var)
        if np.any(np.isfinite(arr)):
            cols.append(arr)
            names.append(var.name)
    if not cols:
        return np.empty((len(data), 0), dtype=float), []
    return np.asarray(np.column_stack(cols), dtype=float), names


def _descriptor_attribute_vars(data: Table, target_var=None) -> list:
    out = []
    for var in data.domain.attributes:
        if target_var is not None and var.name == target_var.name:
            continue
        if not isinstance(var, ContinuousVariable):
            continue
        if _norm_name(var.name) in LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES:
            continue
        out.append(var)
    return out


def _smiles_column_values(data: Table) -> list[str]:
    var = find_smiles_var(data)
    if var is None:
        return []
    col = data.get_column(var)
    values = []
    for value in col:
        if value is None:
            values.append("")
        else:
            text = str(value).strip()
            # Orange StringVariable stores strings directly; defensive for encoded values.
            values.append("" if text.lower() == "nan" else text)
    return values


def _rdkit_descriptor_row(smiles: str) -> list[float]:
    if Chem is None or Descriptors is None:
        raise ValueError("No descriptor columns found and RDKit is not available to compute all-in-one QSAR descriptors.")
    mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol
    if mol is None:
        return [np.nan] * len(RDKit_DESCRIPTOR_NAMES)
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


def _compute_rdkit_descriptor_matrix(data: Table) -> tuple[np.ndarray, list[str]]:
    smiles_values = _smiles_column_values(data)
    if not smiles_values:
        raise ValueError(
            "No usable descriptor attributes were found and no SMILES column was found for automatic descriptor calculation. "
            "Connect QSAR Dataset Builder output with SMILES metas, or compute descriptors before QSAR Regression."
        )
    X = np.asarray([_rdkit_descriptor_row(smiles) for smiles in smiles_values], dtype=float)
    return X, list(RDKit_DESCRIPTOR_NAMES)


def _finite_unique_count(values: np.ndarray) -> int:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0
    return int(np.unique(finite).size)


def clean_qsar_descriptor_matrix(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    remove_constant: bool = True,
) -> tuple[np.ndarray, list[str], dict]:
    """Return a numerically safer QSAR descriptor matrix and cleanup metadata.

    The modeling workflow should not treat audit/provenance columns or fully
    empty descriptor columns as valid descriptors. This helper keeps only
    columns that contain at least one finite value and, by default, removes
    constant/near-constant columns that cannot explain target variance.
    """
    X_arr = np.asarray(X, dtype=float)
    names = list(feature_names)
    if X_arr.ndim != 2:
        X_arr = X_arr.reshape(len(X_arr), -1)

    if X_arr.shape[1] != len(names):
        names = [f"descriptor_{i + 1}" for i in range(X_arr.shape[1])]

    finite_col_mask = np.any(np.isfinite(X_arr), axis=0) if X_arr.size else np.zeros(X_arr.shape[1], dtype=bool)
    unique_counts = np.asarray([_finite_unique_count(X_arr[:, j]) for j in range(X_arr.shape[1])], dtype=int)
    constant_mask = unique_counts <= 1
    keep_mask = finite_col_mask.copy()
    if remove_constant:
        keep_mask &= ~constant_mask

    removed_all_missing = [name for name, keep, finite in zip(names, keep_mask, finite_col_mask) if not finite]
    removed_constant = [
        name for name, keep, finite, const in zip(names, keep_mask, finite_col_mask, constant_mask)
        if finite and const and remove_constant
    ]

    if not np.any(keep_mask):
        # Keep finite non-empty columns as a fallback, otherwise keep the matrix
        # untouched so the caller can raise a clear error. This avoids turning a
        # tiny toy dataset with one constant descriptor into a shape error.
        fallback = finite_col_mask
        if np.any(fallback):
            keep_mask = fallback
            removed_constant = []

    X_clean = X_arr[:, keep_mask] if np.any(keep_mask) else X_arr[:, :0]
    names_clean = [name for name, keep in zip(names, keep_mask) if keep]
    cleanup = {
        "input_descriptor_count": int(X_arr.shape[1]),
        "descriptor_count": int(X_clean.shape[1]),
        "removed_all_missing_count": int(len(removed_all_missing)),
        "removed_constant_count": int(len(removed_constant)),
        "removed_all_missing": removed_all_missing[:30],
        "removed_constant": removed_constant[:30],
    }
    return X_clean, names_clean, cleanup




def _feature_relevance_scores_for_cap(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute lightweight descriptor ranking scores for QSAR feature capping.

    The score is |Pearson r(descriptor, target)| when possible and falls back
    to finite variance. It is intentionally loop-based to avoid allocating a
    large dense normalized copy of high-dimensional descriptor matrices.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_features = int(X_arr.shape[1]) if X_arr.ndim == 2 else 0
    scores = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        x = X_arr[:, j]
        mask = np.isfinite(x) & np.isfinite(y_arr)
        if np.count_nonzero(mask) >= 3:
            xv = x[mask]
            yv = y_arr[mask]
            xstd = float(np.std(xv))
            ystd = float(np.std(yv))
            if xstd > 0 and ystd > 0:
                corr = np.corrcoef(xv, yv)[0, 1]
                if np.isfinite(corr):
                    scores[j] = abs(float(corr))
                    continue
        finite = x[np.isfinite(x)]
        if finite.size > 1:
            var = float(np.nanvar(finite))
            scores[j] = var if np.isfinite(var) else 0.0
    return scores


def cap_qsar_descriptor_matrix(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    max_features: int,
) -> tuple[np.ndarray, list[str], dict]:
    """Apply a final, memory-safe QSAR feature cap before model training.

    Descriptor Pre-selector should normally do this upstream. This extra guard
    protects Orange from accidentally sending thousands of columns into CV,
    grid-search, and diagnostic plots.
    """
    X_arr = np.asarray(X, dtype=float)
    names = list(feature_names)
    max_features = int(max_features or 0)
    if max_features <= 0 or X_arr.ndim != 2 or X_arr.shape[1] <= max_features:
        return X_arr, names, {
            "qsar_cap_applied": False,
            "qsar_cap_limit": max_features,
            "removed_qsar_cap_count": 0,
            "removed_qsar_cap": [],
        }

    scores = _feature_relevance_scores_for_cap(X_arr, y)
    # Keep the best max_features by score, then restore original column order.
    order = np.argsort(scores, kind="mergesort")[::-1]
    keep_idx = np.sort(order[:max_features])
    keep_mask = np.zeros(X_arr.shape[1], dtype=bool)
    keep_mask[keep_idx] = True
    removed = [name for name, keep in zip(names, keep_mask) if not keep]
    kept_names = [name for name, keep in zip(names, keep_mask) if keep]
    return X_arr[:, keep_mask], kept_names, {
        "qsar_cap_applied": True,
        "qsar_cap_limit": max_features,
        "removed_qsar_cap_count": int(len(removed)),
        "removed_qsar_cap": removed[:30],
    }



def _make_safe_regressor(algo_class: type):
    """Instantiate an estimator with Orange-safe defaults."""
    model = algo_class()
    if isinstance(model, (Lasso, ElasticNet)):
        model.set_params(max_iter=10000, random_state=42)
    elif isinstance(model, RandomForestRegressor):
        model.set_params(n_jobs=1, random_state=42)
    elif isinstance(model, GradientBoostingRegressor):
        model.set_params(random_state=42)
    elif isinstance(model, DecisionTreeRegressor):
        model.set_params(random_state=42)
    return model


def _build_modeling_pipeline(
    *,
    algo_class: type,
    imputation_method: int,
    normalization_method: int,
    enable_feature_selection: bool,
    num_features: int,
    n_available_features: int,
    is_classification: bool = False,
) -> Pipeline:
    """Build a fresh sklearn pipeline with safe preprocessing and estimator defaults."""
    steps = []
    if imputation_method != 0:
        strat = {1: "mean", 2: "median", 3: "most_frequent"}.get(imputation_method, "mean")
        steps.append(("imputer", SimpleImputer(strategy=strat)))
    else:
        # Mordred-style descriptor matrices often contain NaNs; keep a fallback
        # imputer even when the UI is set to "None".
        steps.append(("imputer", SimpleImputer(strategy="mean")))

    if normalization_method == 1:
        steps.append(("scaler", StandardScaler()))
    elif normalization_method == 2:
        steps.append(("scaler", MinMaxScaler()))

    if enable_feature_selection:
        score_func = mutual_info_classif if is_classification else safe_f_regression
        k_features = max(1, min(int(num_features), int(n_available_features))) if n_available_features else int(num_features)
        steps.append(("feature_selection", SelectKBest(score_func=score_func, k=k_features)))

    steps.append(("regressor", _make_safe_regressor(algo_class)))
    return Pipeline(steps)


def _auto_qsar_candidates() -> list[tuple[str, type]]:
    """Small, stable one-click model set for QSAR/QSPR regression."""
    return [
        ("Random Forest", RandomForestRegressor),
        ("Gradient Boosting", GradientBoostingRegressor),
        ("Ridge Regression", Ridge),
        ("Elastic Net", ElasticNet),
        ("Support Vector Regression", SVR),
        ("PLS Regression", PLSRegression),
    ]


def _run_auto_qsar_model_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: QSARRunConfig,
    cv_folds: int,
    scoring: str,
) -> tuple[Pipeline, float, str, Table | None, str]:
    """Evaluate a compact model panel and return the best fitted pipeline."""
    rows = []
    best_score = -np.inf
    best_name = ""
    best_pipe: Pipeline | None = None

    for model_name, algo_class in _auto_qsar_candidates():
        # PLS cannot use more components than samples/features; keep the default
        # simple by setting a safe value on the constructed estimator.
        pipe = _build_modeling_pipeline(
            algo_class=algo_class,
            imputation_method=config.imputation_method,
            normalization_method=config.normalization_method,
            enable_feature_selection=config.enable_feature_selection,
            num_features=config.num_features,
            n_available_features=int(X_train.shape[1]),
            is_classification=False,
        )
        if algo_class is PLSRegression:
            n_comp = max(1, min(2, int(X_train.shape[1]), max(1, int(len(y_train)) - 1)))
            pipe.named_steps["regressor"].set_params(n_components=n_comp)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=1, error_score=np.nan)
            mean_score = float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else float("nan")
            std_score = float(np.nanstd(scores)) if np.any(np.isfinite(scores)) else float("nan")
        except Exception as exc:
            mean_score = float("nan")
            std_score = float("nan")
            rows.append({
                "rank": "",
                "model": model_name,
                "cv_score_mean": mean_score,
                "cv_score_std": std_score,
                "selected": 0,
                "status": f"failed: {exc}",
            })
            continue

        rows.append({
            "rank": "",
            "model": model_name,
            "cv_score_mean": mean_score,
            "cv_score_std": std_score,
            "selected": 0,
            "status": "ok" if np.isfinite(mean_score) else "failed/no finite CV score",
        })
        if np.isfinite(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_name = model_name
            best_pipe = pipe

    if best_pipe is None:
        raise ValueError("Auto QSAR failed: no candidate model produced a finite CV score.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        best_pipe.fit(X_train, y_train)

    # Rank successful models; selected gets flag 1.
    ok_rows = [r for r in rows if np.isfinite(float(r["cv_score_mean"]))]
    ranked_names = {id(r): i + 1 for i, r in enumerate(sorted(ok_rows, key=lambda r: float(r["cv_score_mean"]), reverse=True))}
    for r in rows:
        r["rank"] = ranked_names.get(id(r), "")
        r["selected"] = 1 if r["model"] == best_name else 0
    table = records_to_orange_table(rows, name="Auto QSAR Model Ranking") if rows else None
    info = f"Auto QSAR selected {best_name}; best CV {scoring}: {best_score:.3f}\n"
    return best_pipe, best_score, best_name, table, info


def build_applicability_domain_table(result: dict) -> Table | None:
    """Compute a Williams/leverage and kNN-distance AD table for train/test/external rows."""
    if not result or result.get("pipeline") is None:
        return None
    if result.get("is_classification"):
        return None
    try:
        pipeline = result["pipeline"]
        X_train_t = np.asarray(_transform_features(pipeline, result["X_train"]), dtype=float)
        y_train = np.asarray(result["y_train"], dtype=float)
        train_pred = np.asarray(pipeline.predict(result["X_train"]), dtype=float).ravel()
        n_train, p_feat = X_train_t.shape
        if n_train < 3 or p_feat < 1:
            return None

        X_aug = np.column_stack([np.ones(n_train), X_train_t])
        xtx_inv = np.linalg.pinv(X_aug.T @ X_aug)
        h_star = float(3.0 * (p_feat + 1) / max(n_train, 1))

        k = max(1, min(5, n_train))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_train_t)
        train_dist, _ = nn.kneighbors(X_train_t)
        # For training points, the first neighbor is usually itself. Exclude it when possible.
        train_knn = train_dist[:, 1:].mean(axis=1) if train_dist.shape[1] > 1 else train_dist.mean(axis=1)
        dist_threshold = float(np.nanmean(train_knn) + 3.0 * np.nanstd(train_knn)) if train_knn.size else float("nan")

        rows: list[dict] = []

        def add_dataset(dataset: str, X_raw: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
            X_t = np.asarray(_transform_features(pipeline, X_raw), dtype=float)
            X_t_aug = np.column_stack([np.ones(X_t.shape[0]), X_t])
            leverage = np.sum((X_t_aug @ xtx_inv) * X_t_aug, axis=1)
            dist, _ = nn.kneighbors(X_t)
            knn = dist[:, 1:].mean(axis=1) if dataset == "train" and dist.shape[1] > 1 else dist.mean(axis=1)
            for i, (actual, predicted, lev, d) in enumerate(zip(y, pred, leverage, knn)):
                in_lev = bool(np.isfinite(lev) and lev <= h_star)
                in_dist = bool((not np.isfinite(dist_threshold)) or d <= dist_threshold)
                rows.append({
                    "dataset": dataset,
                    "row_index": int(i),
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "residual": float(actual - predicted),
                    "abs_residual": float(abs(actual - predicted)),
                    "ad_leverage": float(lev),
                    "ad_leverage_threshold": h_star,
                    "ad_knn_distance": float(d),
                    "ad_distance_threshold": dist_threshold,
                    "ad_in_domain": int(in_lev and in_dist),
                    "ad_outlier": int((not in_lev) or (not in_dist)),
                })

        add_dataset("train", result["X_train"], y_train, train_pred)
        test_pred = np.asarray(pipeline.predict(result["X_test"]), dtype=float).ravel()
        add_dataset("test", result["X_test"], np.asarray(result["y_test"], dtype=float), test_pred)
        if result.get("X_ext") is not None and result.get("y_ext") is not None:
            ext_pred = np.asarray(pipeline.predict(result["X_ext"]), dtype=float).ravel()
            add_dataset("external", result["X_ext"], np.asarray(result["y_ext"], dtype=float), ext_pred)
        return records_to_orange_table(rows, name="QSAR Applicability Domain") if rows else None
    except Exception:
        return None

def build_qsar_modeling_summary_table(result: dict) -> Table | None:
    """Build a compact modeling audit table for downstream inspection."""
    if not result:
        return None
    cleanup = dict(result.get("descriptor_cleanup") or {})
    records = [
        {
            "section": "dataset",
            "metric": "target_column",
            "value": result.get("target_column", ""),
            "numeric_value": "",
        },
        {
            "section": "dataset",
            "metric": "usable_rows",
            "value": str(result.get("usable_row_count", "")),
            "numeric_value": result.get("usable_row_count", ""),
        },
        {
            "section": "dataset",
            "metric": "removed_rows",
            "value": str(result.get("removed_row_count", "")),
            "numeric_value": result.get("removed_row_count", ""),
        },
        {
            "section": "descriptors",
            "metric": "input_descriptor_count",
            "value": str(cleanup.get("input_descriptor_count", "")),
            "numeric_value": cleanup.get("input_descriptor_count", ""),
        },
        {
            "section": "descriptors",
            "metric": "descriptor_count_used",
            "value": str(cleanup.get("descriptor_count", len(result.get("feature_names", [])))),
            "numeric_value": cleanup.get("descriptor_count", len(result.get("feature_names", []))),
        },
        {
            "section": "descriptors",
            "metric": "removed_all_missing_count",
            "value": str(cleanup.get("removed_all_missing_count", 0)),
            "numeric_value": cleanup.get("removed_all_missing_count", 0),
        },
        {
            "section": "descriptors",
            "metric": "removed_constant_count",
            "value": str(cleanup.get("removed_constant_count", 0)),
            "numeric_value": cleanup.get("removed_constant_count", 0),
        },
        {
            "section": "descriptors",
            "metric": "qsar_cap_limit",
            "value": str(cleanup.get("qsar_cap_limit", "")),
            "numeric_value": cleanup.get("qsar_cap_limit", ""),
        },
        {
            "section": "descriptors",
            "metric": "removed_qsar_cap_count",
            "value": str(cleanup.get("removed_qsar_cap_count", 0)),
            "numeric_value": cleanup.get("removed_qsar_cap_count", 0),
        },
        {
            "section": "model",
            "metric": "cv_score",
            "value": str(result.get("cv_score", "")),
            "numeric_value": result.get("cv_score", ""),
        },
    ]
    if cleanup.get("removed_all_missing"):
        records.append({
            "section": "descriptors",
            "metric": "removed_all_missing_examples",
            "value": ", ".join(cleanup.get("removed_all_missing", [])),
            "numeric_value": "",
        })
    if cleanup.get("removed_constant"):
        records.append({
            "section": "descriptors",
            "metric": "removed_constant_examples",
            "value": ", ".join(cleanup.get("removed_constant", [])),
            "numeric_value": "",
        })
    if cleanup.get("removed_qsar_cap"):
        records.append({
            "section": "descriptors",
            "metric": "removed_qsar_cap_examples",
            "value": ", ".join(cleanup.get("removed_qsar_cap", [])),
            "numeric_value": "",
        })
    return records_to_orange_table(
        records,
        attribute_columns=["numeric_value"],
        meta_columns=["section", "metric", "value"],
        name="QSAR Modeling Summary",
    )


def prepare_qsar_model_matrix(data: Table, *, feature_names: Optional[Sequence[str]] = None) -> dict:
    """Prepare X/y for the all-in-one QSAR regression widget.

    Robust Orange-role handling:
    - target is read from class_var first, then activity-like attributes/metas;
    - descriptor attributes are preferred;
    - if descriptor attributes are absent/empty, numeric metas are accepted as
      a rescue path after Select Columns or CSV import;
    - if no numeric descriptors are available, a compact RDKit descriptor panel
      is computed from SMILES.
    """
    if data is None or len(data) == 0:
        raise ValueError("No rows are available for QSAR regression.")

    target_var = data.domain.class_var or _target_candidate_var(data)
    if target_var is None:
        attr_names = [var.name for var in data.domain.attributes]
        meta_names = [var.name for var in data.domain.metas]
        class_names = [var.name for var in data.domain.class_vars]
        raise ValueError(
            "No numeric target variable found. Expected a class variable or a pActivity/activity-like column.\n"
            f"Class variables: {class_names or 'none'}\n"
            f"Attributes: {attr_names[:20] or 'none'}\n"
            f"Metas: {meta_names[:20] or 'none'}"
        )

    y = _column_as_float(data, target_var)

    generated_descriptors = False

    # Preferred path: descriptors are continuous Orange attributes.
    attr_vars = _descriptor_attribute_vars(data, target_var=target_var)

    # If the caller requests a fixed feature set, use it if available among
    # attributes. Otherwise try numeric metas before falling back to RDKit.
    if feature_names is not None:
        by_name = {var.name: var for var in attr_vars}
        if all(name in by_name for name in feature_names):
            attr_vars = [by_name[name] for name in feature_names]
            X = np.asarray(np.column_stack([_column_as_float(data, var) for var in attr_vars]), dtype=float)
            names = [var.name for var in attr_vars]
        else:
            candidate_vars = [
                var for var in list(data.domain.attributes) + list(data.domain.metas)
                if var.name in set(feature_names)
                and var.name != target_var.name
                and _norm_name(var.name) not in LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES
            ]
            X, names = _numeric_columns_from_vars(data, candidate_vars)
            if len(names) != len(feature_names):
                X, names = _compute_rdkit_descriptor_matrix(data)
                generated_descriptors = True
        finite_y = np.isfinite(y)
        if not np.any(finite_y):
            raise ValueError(f"Target column '{target_var.name}' contains no numeric values.")
        if X.shape[1] == 0:
            raise ValueError("No descriptor columns are available for QSAR regression.")
        return {
            "X": X,
            "y": y,
            "metas": np.array(data.metas),
            "target_var": target_var,
            "feature_names": names,
            "generated_descriptors": generated_descriptors,
        }

    if attr_vars:
        X = np.asarray(np.column_stack([_column_as_float(data, var) for var in attr_vars]), dtype=float)
        names = [var.name for var in attr_vars]
    else:
        # Rescue path for Select Columns / CSV workflows where descriptors
        # accidentally became metas. Keep only numeric, non-leaky meta columns.
        meta_vars = [
            var for var in data.domain.metas
            if var.name != target_var.name
            and _norm_name(var.name) not in LEAKY_OR_NON_DESCRIPTOR_ATTRIBUTE_NAMES
            and _norm_name(var.name) not in TARGET_COLUMN_CANDIDATES
            and _norm_name(var.name) not in {"smiles", "smile", "canonicalsmiles", "canonical_smiles"}
        ]
        X, names = _numeric_columns_from_vars(data, meta_vars)
        if X.shape[1] == 0:
            X, names = _compute_rdkit_descriptor_matrix(data)
            generated_descriptors = True

    finite_y = np.isfinite(y)
    if not np.any(finite_y):
        raise ValueError(f"Target column '{target_var.name}' contains no numeric values.")
    if X.shape[1] == 0:
        raise ValueError("No descriptor columns are available for QSAR regression.")

    # Do not accept columns that are all NaN as descriptors.
    finite_descriptor_cols = np.any(np.isfinite(X), axis=0)
    if not np.any(finite_descriptor_cols):
        raise ValueError(
            "Descriptor columns were found, but all descriptor values are missing/non-numeric. "
            "Check Select Columns: descriptor columns must remain numeric attributes, or keep SMILES for automatic RDKit descriptors."
        )
    if not np.all(finite_descriptor_cols):
        X = X[:, finite_descriptor_cols]
        names = [name for name, keep in zip(names, finite_descriptor_cols) if keep]

    return {
        "X": X,
        "y": y,
        "metas": np.array(data.metas),
        "target_var": target_var,
        "feature_names": names,
        "generated_descriptors": generated_descriptors,
    }


def _result_domain(source_data: Table, feature_names: Sequence[str], target_var, *, is_classification: bool = False) -> Domain:
    attributes = [ContinuousVariable(str(name)) for name in feature_names]
    pred_var = ContinuousVariable("Predicted") if not is_classification else DiscreteVariable("Predicted")
    class_vars = [ContinuousVariable(target_var.name)] if isinstance(target_var, ContinuousVariable) else [target_var]
    return Domain(attributes + [pred_var], class_vars, source_data.domain.metas)


def build_report_html(
    *,
    model_name: str,
    total_descriptors: int,
    descriptors_used: int,
    cv_score: Optional[float],
    train_metrics: dict,
    test_metrics: dict,
    external_metrics: dict,
) -> str:
    cv_text = f"{cv_score:.3f}" if cv_score is not None else "N/A"
    metrics_list = ["R²", "RMSE", "MAE", "Median AE", "Explained Variance"]

    html = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; font-size: 12pt; color: #333; }}
          h2 {{ color: #444; }}
          table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
          th, td {{ border: 1px solid #ccc; padding: 5px; text-align: center; }}
          th {{ background-color: #f0f0f0; }}
        </style>
      </head>
      <body>
        <h2>Model Report</h2>
        <p><b>Model:</b> {model_name}</p>
        <p><b>Total Descriptors:</b> {total_descriptors}</p>
        <p><b>Descriptors Used:</b> {descriptors_used}</p>
        <p><b>CV R²:</b> {cv_text}</p>
        <h3>Metrics</h3>
        <table>
          <tr>
            <th>Metric</th>
            <th>Training</th>
            <th>Test</th>
            <th>External</th>
          </tr>
    """
    for metric in metrics_list:
        train_val = f"{train_metrics[metric]:.3f}" if metric in train_metrics else "N/A"
        test_val = f"{test_metrics[metric]:.3f}" if metric in test_metrics else "N/A"
        ext_val = f"{external_metrics[metric]:.3f}" if external_metrics and metric in external_metrics else "N/A"
        html += f"""
          <tr>
            <td>{metric}</td>
            <td>{train_val}</td>
            <td>{test_val}</td>
            <td>{ext_val}</td>
          </tr>
        """
    html += """
        </table>
      </body>
    </html>
    """
    return html


def build_report_context(
    *,
    model_name: str,
    total_descriptors: int,
    descriptors_used: int,
    cv_score: Optional[float],
    train_metrics: dict,
    test_metrics: dict,
    external_metrics: dict,
) -> ReportContext:
    return ReportContext(
        model_name=model_name,
        total_descriptors=total_descriptors,
        descriptors_used=descriptors_used,
        cv_score=cv_score,
        train_metrics=dict(train_metrics),
        test_metrics=dict(test_metrics),
        external_metrics=dict(external_metrics),
    )


def build_report_html_from_context(context: ReportContext) -> str:
    return build_report_html(
        model_name=context.model_name,
        total_descriptors=context.total_descriptors,
        descriptors_used=context.descriptors_used,
        cv_score=context.cv_score,
        train_metrics=context.train_metrics,
        test_metrics=context.test_metrics,
        external_metrics=context.external_metrics,
    )


def build_waiting_status_text(model_name: str) -> str:
    return f"Please wait calculation {model_name} is started"


def build_waiting_report_html() -> str:
    return (
        '<div style="text-align: center; font-weight: bold; font-size: 14pt;">'
        "Please wait, calculation of the QSAR model in progress"
        "</div>"
    )


def build_cancelled_status_text() -> str:
    return "Calculation cancelled."


def build_completed_status_text(model_name: str, performance_text: str) -> str:
    return f"Calculation {model_name} is completed.\n{performance_text}"


def build_error_status_text(error_msg: str) -> str:
    return "Error: " + str(error_msg)


def build_pdf_export_success_status_text() -> str:
    return "PDF Exported Successfully."


def build_pdf_export_empty_status_text() -> str:
    return "No QSAR results available to export."


def build_pdf_export_error_status_text(error_msg: str) -> str:
    return "Error exporting PDF: " + str(error_msg)


def build_pdf_report_text(
    *,
    model_name: str,
    total_descriptors: int,
    descriptors_used: int,
    cv_score: Optional[float],
    train_metrics: dict,
    test_metrics: dict,
    external_metrics: dict,
) -> str:
    cv_info = f"CV R²: {cv_score:.3f}\n\n" if cv_score is not None else "CV R²: N/A\n\n"
    report_text = (
        f"Model: {model_name}\n"
        f"Total Descriptors: {total_descriptors}\n"
        f"Descriptors Used: {descriptors_used}\n"
        f"{cv_info}"
    )

    report_text += "Training Metrics:\n"
    for key, value in train_metrics.items():
        report_text += f"  {key}: {value:.3f}\n"

    report_text += "\nTest Metrics:\n"
    for key, value in test_metrics.items():
        report_text += f"  {key}: {value:.3f}\n"

    report_text += "\nExternal Metrics:\n"
    for key, value in external_metrics.items():
        report_text += f"  {key}: {value:.3f}\n"

    return report_text


def build_pdf_report_text_from_context(context: ReportContext) -> str:
    return build_pdf_report_text(
        model_name=context.model_name,
        total_descriptors=context.total_descriptors,
        descriptors_used=context.descriptors_used,
        cv_score=context.cv_score,
        train_metrics=context.train_metrics,
        test_metrics=context.test_metrics,
        external_metrics=context.external_metrics,
    )


def build_pdf_report_figure_from_context(context: ReportContext) -> Figure:
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    report_text = build_pdf_report_text_from_context(context)
    ax.text(0, 1, report_text, va="top", ha="left", fontsize=10, wrap=True)
    return fig


def collect_pdf_export_figures(*figures) -> tuple:
    return tuple(fig for fig in figures if fig is not None)


def build_compound_previews(selected_table: Optional[Table], *, max_preview: int = 12) -> list[CompoundPreview]:
    if selected_table is None or len(selected_table) == 0:
        return []

    smiles_var = find_smiles_var(selected_table)
    if smiles_var is None:
        return []

    name_var = find_name_var(selected_table)
    smiles_col = selected_table.get_column(smiles_var)
    names_col = selected_table.get_column(name_var) if name_var is not None else None

    previews: list[CompoundPreview] = []
    for i in range(min(len(selected_table), max_preview)):
        smiles = "" if smiles_col[i] is None else str(smiles_col[i]).strip()
        if not smiles:
            continue

        mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol
        if mol is None:
            continue

        img = Draw.MolToImage(mol, size=(150, 110))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        title = ""
        if names_col is not None and names_col[i] is not None:
            title = str(names_col[i]).strip()
        if not title:
            title = f"Row {i + 1}"

        previews.append(CompoundPreview(title=title, png_bytes=buf.getvalue()))

    return previews


def rectangle_selection_indices(preds, ys, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    preds_arr = np.asarray(preds, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    x_min, x_max = sorted([float(x0), float(x1)])
    y_min, y_max = sorted([float(y0), float(y1)])
    mask = (preds_arr >= x_min) & (preds_arr <= x_max) & (ys_arr >= y_min) & (ys_arr <= y_max)
    return np.flatnonzero(mask)


def lasso_selection_indices(preds, ys, vertices) -> np.ndarray:
    if not vertices:
        return np.array([], dtype=int)
    preds_arr = np.asarray(preds, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    points = np.column_stack([preds_arr, ys_arr])
    mask = Path(vertices).contains_points(points)
    return np.flatnonzero(mask)


def selection_overlay_offsets(preds, y, residuals, selected_idx) -> tuple[np.ndarray, np.ndarray]:
    preds_arr = np.asarray(preds, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    residuals_arr = np.asarray(residuals, dtype=float)
    idx = np.asarray(selected_idx, dtype=int)
    if idx.size == 0:
        empty = np.empty((0, 2))
        return empty, empty
    left_offsets = np.column_stack([preds_arr[idx], y_arr[idx]])
    right_offsets = np.column_stack([preds_arr[idx], residuals_arr[idx]])
    return left_offsets, right_offsets


def selection_status_text(model_name: str, dataset_type: str, count: int) -> str:
    return (
        f"Calculation {model_name} is completed.\n"
        f"Selected {int(count)} compounds from {dataset_type} diagnostics."
    )


def build_selection_gallery_payload(
    selected_table: Optional[Table],
    dataset_type: str,
    *,
    max_preview: int = 12,
) -> SelectionGalleryPayload:
    if selected_table is None or len(selected_table) == 0:
        return SelectionGalleryPayload(
            placeholder_text=f"No compounds selected in {dataset_type} diagnostics.",
            previews=tuple(),
            more_count=0,
        )

    previews = build_compound_previews(selected_table, max_preview=max_preview)
    if not previews:
        return SelectionGalleryPayload(
            placeholder_text=(
                f"Selected {len(selected_table)} rows from {dataset_type}, but no valid molecules could be rendered."
            ),
            previews=tuple(),
            more_count=0,
        )

    shown = len(previews)
    more_count = max(0, len(selected_table) - shown)
    return SelectionGalleryPayload(
        placeholder_text=None,
        previews=tuple(previews),
        more_count=more_count,
    )


def build_selection_publish_payload(
    *,
    model_name: str,
    dataset_type: str,
    table: Table,
    selected_idx,
    max_preview: int = 12,
) -> SelectionPublishPayload:
    idx = np.asarray(selected_idx, dtype=int)
    selected_table = table[idx.tolist()] if idx.size else table[:0]
    gallery = build_selection_gallery_payload(selected_table, dataset_type, max_preview=max_preview)
    return SelectionPublishPayload(
        selected_table=selected_table,
        gallery=gallery,
        status_text=selection_status_text(model_name, dataset_type, int(idx.size)),
    )


def prepare_diagnostic_plot_data(X, y, pipeline, *, is_classification: bool = False) -> DiagnosticPlotData:
    preds = np.asarray(pipeline.predict(X), dtype=float)
    actuals = np.asarray(y, dtype=float)

    if not is_classification:
        residuals = actuals - preds
        threshold = 2 * np.std(residuals)
        inlier_mask = np.abs(residuals) <= threshold
        outlier_mask = np.abs(residuals) > threshold
    else:
        residuals = (actuals != preds).astype(float)
        inlier_mask = np.ones_like(preds, dtype=bool)
        outlier_mask = np.zeros_like(preds, dtype=bool)

    return DiagnosticPlotData(
        preds=preds,
        actuals=actuals,
        residuals=residuals,
        inlier_mask=inlier_mask,
        outlier_mask=outlier_mask,
        is_classification=bool(is_classification),
    )


def build_diagnostic_plot_spec(diagnostic: DiagnosticPlotData) -> DiagnosticPlotSpec:
    diagonal_min = float(np.min(diagnostic.preds)) if diagnostic.preds.size else 0.0
    diagonal_max = float(np.max(diagnostic.preds)) if diagnostic.preds.size else 1.0

    if not diagnostic.is_classification:
        left_series = [
            DiagnosticSeries(
                x=diagnostic.preds[diagnostic.inlier_mask],
                y=diagnostic.actuals[diagnostic.inlier_mask],
                color="blue",
                label="Inliers",
            )
        ]
        right_series = [
            DiagnosticSeries(
                x=diagnostic.preds[diagnostic.inlier_mask],
                y=diagnostic.residuals[diagnostic.inlier_mask],
                color="blue",
                label="Inliers",
            )
        ]
        if np.any(diagnostic.outlier_mask):
            left_series.append(
                DiagnosticSeries(
                    x=diagnostic.preds[diagnostic.outlier_mask],
                    y=diagnostic.actuals[diagnostic.outlier_mask],
                    color="red",
                    label="Outliers",
                )
            )
            right_series.append(
                DiagnosticSeries(
                    x=diagnostic.preds[diagnostic.outlier_mask],
                    y=diagnostic.residuals[diagnostic.outlier_mask],
                    color="red",
                    label="Outliers",
                )
            )
        return DiagnosticPlotSpec(
            left_series=tuple(left_series),
            right_series=tuple(right_series),
            diagonal_min=diagonal_min,
            diagonal_max=diagonal_max,
            left_title="Predicted vs Actual",
            left_xlabel="Predicted",
            left_ylabel="Actual",
            right_title="Residuals vs Predicted",
            right_xlabel="Predicted",
            right_ylabel="Residuals",
            show_legends=True,
        )

    return DiagnosticPlotSpec(
        left_series=(
            DiagnosticSeries(
                x=diagnostic.preds,
                y=diagnostic.actuals,
                color="green",
                label="Observations",
            ),
        ),
        right_series=(
            DiagnosticSeries(
                x=diagnostic.preds,
                y=diagnostic.residuals,
                color="green",
                label="Observations",
            ),
        ),
        diagonal_min=diagonal_min,
        diagonal_max=diagonal_max,
        left_title="Predicted vs Actual",
        left_xlabel="Predicted",
        left_ylabel="Actual",
        right_title="Misclassifications (1 if error)",
        right_xlabel="Predicted",
        right_ylabel="Error Indicator",
        show_legends=False,
    )


def diagnostic_payloads_from_result(result: dict, *, include_external: bool = False) -> list[DiagnosticDatasetPayload]:
    payloads = [
        DiagnosticDatasetPayload(
            dataset_type="train",
            X=result["X_train"],
            y=result["y_train"],
            pipeline=result["pipeline"],
            is_classification=result["is_classification"],
            result_table=result["train_table"],
        ),
        DiagnosticDatasetPayload(
            dataset_type="test",
            X=result["X_test"],
            y=result["y_test"],
            pipeline=result["pipeline"],
            is_classification=result["is_classification"],
            result_table=result["test_table"],
        ),
    ]

    if include_external and result.get("external_table") is not None:
        payloads.append(
            DiagnosticDatasetPayload(
                dataset_type="external",
                X=result["X_ext"],
                y=result["y_ext"],
                pipeline=result["pipeline"],
                is_classification=result["is_classification"],
                result_table=result["external_table"],
            )
        )

    return payloads


def build_feature_inspection_payload(result: dict, *, model_name: str) -> FeatureInspectionPayload:
    pipeline = result.get("pipeline")
    feature_names = list(result.get("feature_names", []))

    if pipeline is None or not feature_names:
        return FeatureInspectionPayload(
            available=False,
            message_html="No feature information available.",
            value_label="value",
            names=tuple(),
            values=None,
            ses=None,
            ts=None,
            ps=None,
            vifs=None,
            chart_names=tuple(),
            chart_values=None,
            chart_colors=tuple(),
            chart_title="",
            subtitle="",
            tab_title="Features",
        )

    names = list(feature_names)
    if "feature_selection" in pipeline.named_steps:
        try:
            mask = pipeline.named_steps["feature_selection"].get_support()
            names = [n for n, keep in zip(names, mask) if keep]
        except Exception:
            pass

    estimator = pipeline.named_steps.get("regressor", pipeline[-1])
    values: np.ndarray | None = None
    value_label = "value"

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_).ravel()
        if len(coef) == len(names):
            values = coef
            value_label = "Coefficient"
    if values is None and hasattr(estimator, "feature_importances_"):
        imp = np.asarray(estimator.feature_importances_).ravel()
        if len(imp) == len(names):
            values = imp
            value_label = "Importance"

    if values is None:
        return FeatureInspectionPayload(
            available=True,
            message_html=(
                f"<b>{model_name}</b> does not expose coefficients or "
                "feature importances directly.<br>"
                "Selected descriptors are listed in the <i>Descriptor Coefficients</i> output."
            ),
            value_label=value_label,
            names=tuple(names),
            values=None,
            ses=None,
            ts=None,
            ps=None,
            vifs=None,
            chart_names=tuple(),
            chart_values=None,
            chart_colors=tuple(),
            chart_title="",
            subtitle=f"{len(names)} selected descriptors (no {value_label.lower()} values for this model type)",
            tab_title=f"Features ({len(names)})",
        )

    order = np.argsort(np.abs(values))[::-1]
    names = [names[i] for i in order]
    values = np.asarray(values[order], dtype=float)

    coef_stats = result.get("coef_stats")
    ses_s = ts_s = ps_s = None
    if coef_stats is not None:
        beta_d = coef_stats["beta"][1:]
        if len(beta_d) == len(names):
            ses_s = np.asarray(coef_stats["se"][1:][order], dtype=float)
            ts_s = np.asarray(coef_stats["t"][1:][order], dtype=float)
            ps_s = np.asarray(coef_stats["p"][1:][order], dtype=float)

    vifs_all = result.get("vifs")
    vifs_s = np.asarray(vifs_all[order], dtype=float) if (vifs_all is not None and len(vifs_all) == len(names)) else None

    top_n = min(30, len(names))
    chart_names = tuple(names[:top_n])
    chart_values = np.asarray(values[:top_n], dtype=float)
    chart_colors = tuple("#16a34a" if v >= 0 else "#dc2626" for v in chart_values)

    total_feature_count = len(feature_names)
    suffix = f" ({total_feature_count - len(names)} dropped by SelectKBest)" if len(names) < total_feature_count else ""

    return FeatureInspectionPayload(
        available=True,
        message_html="",
        value_label=value_label,
        names=tuple(names),
        values=values,
        ses=ses_s,
        ts=ts_s,
        ps=ps_s,
        vifs=vifs_s,
        chart_names=chart_names,
        chart_values=chart_values,
        chart_colors=chart_colors,
        chart_title=(
            f"{model_name} — {value_label}s  "
            f"(top {top_n} of {len(names)} {'selected ' if len(names) < total_feature_count else ''}descriptors)"
        ),
        subtitle=f"All {len(names)} descriptors — sorted by |{value_label.lower()}|",
        tab_title=f"Features ({len(names)}){suffix}",
    )


def run_qsar_regression(
    data: Table,
    external_data: Optional[Table],
    config: QSARRunConfig,
    *,
    interruption_requested: Optional[Callable[[], bool]] = None,
):
    def cancelled() -> bool:
        return bool(interruption_requested and interruption_requested())

    if cancelled():
        return None

    prepared = prepare_qsar_model_matrix(data)
    X_all = prepared["X"]
    y_all = prepared["y"]
    metas_all = prepared["metas"]
    feature_names = prepared["feature_names"]
    target_var = prepared["target_var"]
    generated_descriptors = bool(prepared["generated_descriptors"])

    finite_y = np.isfinite(y_all)
    # A row is usable if: target is finite AND at least one descriptor value is finite.
    # Rows with some NaN descriptors are kept here — the imputation step in the pipeline
    # will fill them. Requiring ALL descriptors to be finite would drop every row when
    # descriptor sets like Mordred contain even a single NaN per molecule.
    if X_all.ndim == 2 and X_all.shape[1]:
        has_any_finite_x = np.any(np.isfinite(X_all), axis=1)
        finite_x_rows = np.all(np.isfinite(X_all), axis=1)  # for diagnostic message only
    else:
        has_any_finite_x = np.zeros(len(y_all), dtype=bool)
        finite_x_rows = has_any_finite_x
    finite_rows = finite_y & has_any_finite_x
    if np.count_nonzero(finite_rows) < 3:
        raise ValueError(
            "Too few rows with valid target and descriptor values for QSAR regression "
            f"({np.count_nonzero(finite_rows)} usable rows).\n"
            f"Rows with numeric target '{target_var.name}': {np.count_nonzero(finite_y)} / {len(y_all)}.\n"
            f"Rows with at least one finite descriptor: {np.count_nonzero(has_any_finite_x)} / {len(y_all)}.\n"
            f"Rows with all descriptors finite: {np.count_nonzero(finite_x_rows)} / {len(y_all)}.\n"
            f"Descriptor columns used: {len(feature_names)} ({', '.join(feature_names[:8])}"
            + ("..." if len(feature_names) > 8 else "")
            + ").\n"
            "Fix: in Select Columns, keep pActivity as Target/class variable and keep descriptor columns as Features. "
            "Alternatively keep a SMILES column so QSAR Regression can compute RDKit descriptors automatically."
        )
    original_row_count = int(len(y_all))
    usable_row_count = int(np.count_nonzero(finite_rows))
    X_all = X_all[finite_rows]
    y_all = y_all[finite_rows]
    metas_all = metas_all[finite_rows]

    X_all, feature_names, descriptor_cleanup = clean_qsar_descriptor_matrix(X_all, feature_names)
    X_all, feature_names, cap_cleanup = cap_qsar_descriptor_matrix(
        X_all,
        y_all,
        feature_names,
        max_features=getattr(config, "max_model_features", 1000),
    )
    descriptor_cleanup.update(cap_cleanup)
    descriptor_cleanup["descriptor_count"] = int(X_all.shape[1])
    if X_all.shape[1] == 0:
        raise ValueError(
            "No informative descriptor columns remain after removing empty/constant descriptors. "
            "Compute descriptor/fingerprint columns before QSAR or keep a valid SMILES column for RDKit fallback descriptors."
        )

    X_train, X_test, y_train, y_test, metas_train, metas_test = train_test_split(
        X_all,
        y_all,
        metas_all,
        test_size=config.test_size,
        random_state=42,
    )
    if cancelled():
        return None

    algo_name, algo_class = config.algorithms[config.selected_algorithm]
    is_classification = False
    scoring = "r2"
    if algo_name == "Logistic Regression":
        if len(np.unique(y_train)) == 2:
            is_classification = True
            scoring = "accuracy"

    # Avoid CV errors when the user requests more folds than training rows.
    cv_folds = max(2, min(int(config.cv_folds), int(len(y_train))))

    model_ranking_table = None
    auto_mode = bool(getattr(config, "enable_auto_qsar", False)) and not is_classification
    if auto_mode:
        best_pipeline, cv_score, algo_name, model_ranking_table, tuning_info = _run_auto_qsar_model_selection(
            X_train,
            y_train,
            config=config,
            cv_folds=cv_folds,
            scoring=scoring,
        )
        pipeline = best_pipeline
        hp = {}
    else:
        pipeline = _build_modeling_pipeline(
            algo_class=algo_class,
            imputation_method=config.imputation_method,
            normalization_method=config.normalization_method,
            enable_feature_selection=config.enable_feature_selection,
            num_features=config.num_features,
            n_available_features=int(X_train.shape[1]),
            is_classification=is_classification,
        )
        hp = {}
    if (not auto_mode) and config.hyperparameters.strip():
        try:
            hp = json.loads(config.hyperparameters)
        except Exception as exc:  # pragma: no cover - UI surfaced
            raise Exception("Error parsing hyperparameters: " + str(exc)) from exc

    if auto_mode:
        best_pipeline = pipeline
    elif config.tuning_method == 1 and hp:
        tuner = GridSearchCV(
            pipeline,
            param_grid=hp,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            tuner.fit(X_train, y_train)
        if cancelled():
            return None
        best_pipeline = tuner.best_estimator_
        cv_score = tuner.best_score_
        tuning_info = f"Grid Search best CV {scoring}: {cv_score:.3f}\n"
    elif config.tuning_method == 2 and hp:
        tuner = RandomizedSearchCV(
            pipeline,
            param_distributions=hp,
            cv=cv_folds,
            scoring=scoring,
            n_iter=config.n_iter,
            n_jobs=1,
            random_state=42,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            tuner.fit(X_train, y_train)
        if cancelled():
            return None
        best_pipeline = tuner.best_estimator_
        cv_score = tuner.best_score_
        tuning_info = f"Randomized Search best CV {scoring}: {cv_score:.3f}\n"
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring)
        cv_score = np.mean(cv_scores)
        if cancelled():
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            best_pipeline = pipeline.fit(X_train, y_train)
        tuning_info = f"CV {scoring} (no tuning): {cv_score:.3f}\n"

    if cancelled():
        return None

    if not is_classification:
        train_preds = best_pipeline.predict(X_train)
        test_preds = best_pipeline.predict(X_test)
        train_metrics = {
            "R²": r2_score(y_train, train_preds),
            "RMSE": np.sqrt(mean_squared_error(y_train, train_preds)),
            "MAE": mean_absolute_error(y_train, train_preds),
            "Median AE": median_absolute_error(y_train, train_preds),
            "Explained Variance": explained_variance_score(y_train, train_preds),
        }
        test_metrics = {
            "R²": r2_score(y_test, test_preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
            "MAE": mean_absolute_error(y_test, test_preds),
            "Median AE": median_absolute_error(y_test, test_preds),
            "Explained Variance": explained_variance_score(y_test, test_preds),
        }
        performance_text = (
            f"{algo_name}: \n \n {tuning_info}\n"
            f"Train R²: {train_metrics['R²']:.3f}, RMSE: {train_metrics['RMSE']:.3f}, MAE: {train_metrics['MAE']:.3f},"
            f" MedAE: {train_metrics['Median AE']:.3f}, Expl.Var: {train_metrics['Explained Variance']:.3f}\n"
            f"Test R²: {test_metrics['R²']:.3f}, RMSE: {test_metrics['RMSE']:.3f}, MAE: {test_metrics['MAE']:.3f}, "
            f"MedAE: {test_metrics['Median AE']:.3f}, Expl.Var: {test_metrics['Explained Variance']:.3f}\n"
        )
    else:
        train_preds = best_pipeline.predict(X_train)
        test_preds = best_pipeline.predict(X_test)
        test_score = best_pipeline.score(X_test, y_test)
        performance_text = f"{algo_name}: {tuning_info} | Test Accuracy: {test_score:.3f}"
        train_metrics = {}
        test_metrics = {"Accuracy": test_score}

    ext_table = None
    external_metrics = {}
    X_ext = y_ext = None
    if external_data is not None:
        ext_prepared = prepare_qsar_model_matrix(external_data, feature_names=feature_names)
        X_ext = ext_prepared["X"]
        y_ext = ext_prepared["y"]
        metas_ext = ext_prepared["metas"]
        finite_ext = np.isfinite(y_ext) & np.any(np.isfinite(X_ext), axis=1)
        X_ext = X_ext[finite_ext]
        y_ext = y_ext[finite_ext]
        metas_ext = metas_ext[finite_ext]
        ext_preds = best_pipeline.predict(X_ext).reshape(-1, 1)
        ext_domain = _result_domain(external_data, feature_names, ext_prepared["target_var"], is_classification=is_classification)
        ext_table = safe_table_from_numpy(ext_domain, X=np.hstack([X_ext, ext_preds]), Y=y_ext.reshape(-1, 1), metas=metas_ext, name="External Results")
        if not is_classification and len(y_ext) > 1:
            ext_preds_full = best_pipeline.predict(X_ext)
            external_metrics = {
                "R²": r2_score(y_ext, ext_preds_full),
                "RMSE": np.sqrt(mean_squared_error(y_ext, ext_preds_full)),
                "MAE": mean_absolute_error(y_ext, ext_preds_full),
                "Median AE": median_absolute_error(y_ext, ext_preds_full),
                "Explained Variance": explained_variance_score(y_ext, ext_preds_full),
            }

    if cancelled():
        return None

    new_domain = _result_domain(data, feature_names, target_var, is_classification=is_classification)
    train_table = safe_table_from_numpy(new_domain, X=np.hstack([X_train, train_preds.reshape(-1, 1)]), Y=y_train.reshape(-1, 1), metas=metas_train, name="QSAR Train Results")
    test_table = safe_table_from_numpy(new_domain, X=np.hstack([X_test, test_preds.reshape(-1, 1)]), Y=y_test.reshape(-1, 1), metas=metas_test, name="QSAR Test Results")

    # ── Applicability domain, VIF, coef stats, permutation test ──────────
    ad_info: dict = {}
    try:
        if not is_classification:
            X_train_t = _transform_features(best_pipeline, X_train)
            estimator = best_pipeline.named_steps.get("regressor", best_pipeline[-1])
            n_obs, n_feat = X_train_t.shape
            linear_diag_notes = []
            if isinstance(estimator, PLSRegression):
                linear_diag_notes.append("PLS latent-space model: skipped OLS coefficient statistics and VIF diagnostics.")
            elif isinstance(estimator, (Lasso, Ridge, ElasticNet)):
                linear_diag_notes.append(
                    f"{type(estimator).__name__} regularized model: skipped OLS coefficient statistics and VIF diagnostics."
                )
            elif n_feat > 256:
                linear_diag_notes.append(
                    f"Skipped OLS coefficient statistics and VIF diagnostics for {n_feat} features (>256 safety limit)."
                )
            elif n_obs <= (n_feat + 1):
                linear_diag_notes.append(
                    f"Skipped OLS coefficient statistics because training rows ({n_obs}) are not greater than features + intercept ({n_feat + 1})."
                )
            else:
                ad_info["vifs"] = compute_vif(X_train_t)
                coef_stats = compute_coef_stats(X_train_t, y_train, estimator)
                if coef_stats is not None:
                    ad_info["coef_stats"] = coef_stats
            if linear_diag_notes:
                ad_info["linear_diagnostics_note"] = " ".join(linear_diag_notes)
    except Exception:
        pass

    result = {
        "model": best_pipeline,
        "model_name": algo_name,
        "requested_model_name": config.algorithms[config.selected_algorithm][0],
        "auto_qsar_used": bool(auto_mode),
        "train_table": train_table,
        "test_table": test_table,
        "external_table": ext_table,
        "pipeline": best_pipeline,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "is_classification": is_classification,
        "performance_text": performance_text,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "external_metrics": external_metrics,
        "cv_score": cv_score,
        "feature_names": feature_names,
        "generated_descriptors": generated_descriptors,
        "target_column": target_var.name,
        "original_row_count": original_row_count,
        "usable_row_count": usable_row_count,
        "removed_row_count": int(original_row_count - usable_row_count),
        "descriptor_cleanup": descriptor_cleanup,
        "model_ranking_table": model_ranking_table,
        **ad_info,
    }
    if bool(getattr(config, "enable_applicability_domain", True)):
        result["applicability_domain_table"] = build_applicability_domain_table(result)
    else:
        result["applicability_domain_table"] = None
    result["modeling_summary_table"] = build_qsar_modeling_summary_table(result)
    if external_data is not None:
        result["X_ext"] = X_ext
        result["y_ext"] = y_ext

    return result


# ── Statistical diagnostics helpers ──────────────────────────────────────────

def _transform_features(pipeline, X: np.ndarray) -> np.ndarray:
    """Apply all pipeline steps except the final estimator."""
    pre = Pipeline(pipeline.steps[:-1]) if len(pipeline.steps) > 1 else None
    return pre.transform(X) if pre is not None else X


def compute_vif(X_t: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor per feature column."""
    from sklearn.linear_model import LinearRegression as _LR
    n, p = X_t.shape
    if p > 256:
        raise ValueError("VIF is disabled for descriptor spaces wider than 256 features.")
    if p < 2:
        return np.ones(p)
    vifs = np.zeros(p)
    for j in range(p):
        Xo = np.delete(X_t, j, axis=1)
        reg = _LR().fit(Xo, X_t[:, j])
        r2  = float(r2_score(X_t[:, j], reg.predict(Xo)))
        vifs[j] = 1.0 / max(1.0 - r2, 1e-12)
    return vifs


def compute_coef_stats(
    X_t: np.ndarray,
    y: np.ndarray,
    estimator,
) -> dict | None:
    """OLS-style SE / t / p for linear models (coef_ attribute). None for non-linear."""
    if isinstance(estimator, (PLSRegression, Lasso, Ridge, ElasticNet)):
        return None
    from scipy.stats import t as _t_dist
    if not hasattr(estimator, "coef_"):
        return None
    coef = np.asarray(estimator.coef_).ravel()
    n, p = X_t.shape
    if p > 256 or n <= (p + 1):
        return None
    if len(coef) != p:
        return None
    intercept = float(getattr(estimator, "intercept_", 0.0))
    beta = np.concatenate([[intercept], coef])
    X1   = np.column_stack([np.ones(n), X_t])
    y_pred = estimator.predict(X_t)
    resid  = y - y_pred
    dof    = max(n - (p + 1), 1)
    sigma2 = float((resid @ resid) / dof)
    XtX = X1.T @ X1
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(XtX)
    se    = np.sqrt(np.maximum(np.diag(inv) * sigma2, 1e-300))
    tvals = beta / np.maximum(se, 1e-300)
    pvals = 2.0 * (1.0 - _t_dist.cdf(np.abs(tvals), df=dof))
    return {"beta": beta, "se": se, "t": tvals, "p": pvals}


def extract_descriptor_coefficients(pipeline, feature_names: Sequence[str]) -> Table | None:
    """Return an Orange Table with one row per (selected) descriptor and its coefficient or importance.

    Columns: descriptor (string), value (float), abs_value (float).
    The 'value' column is labelled 'coefficient' for linear models and 'importance' for tree models.
    Returns None when the estimator exposes neither coef_ nor feature_importances_.
    """
    names = list(feature_names)

    # Narrow names to those that survived feature selection
    if "feature_selection" in pipeline.named_steps:
        selector = pipeline.named_steps["feature_selection"]
        try:
            mask = selector.get_support()
            names = [n for n, keep in zip(names, mask) if keep]
        except Exception:
            pass

    # Locate the final estimator (named 'regressor' or last step)
    estimator = pipeline.named_steps.get("regressor", pipeline[-1])

    values: np.ndarray | None = None
    col_label = "value"

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_).ravel()
        if len(coef) == len(names):
            values = coef
            col_label = "coefficient"
    if values is None and hasattr(estimator, "feature_importances_"):
        imp = np.asarray(estimator.feature_importances_).ravel()
        if len(imp) == len(names):
            values = imp
            col_label = "importance"

    if values is None:
        return None

    # Sort by |value| descending
    order = np.argsort(np.abs(values))[::-1]
    names_sorted  = [names[i]          for i in order]
    values_sorted = [float(values[i])  for i in order]
    abs_sorted    = [abs(float(values[i])) for i in order]

    domain = Domain(
        [ContinuousVariable(col_label), ContinuousVariable("abs_" + col_label)],
        metas=[StringVariable("descriptor")],
    )
    X = np.column_stack([values_sorted, abs_sorted]).astype(float)
    M = np.array([[n] for n in names_sorted], dtype=object)
    return Table.from_numpy(domain, X=X, Y=None, metas=M)
