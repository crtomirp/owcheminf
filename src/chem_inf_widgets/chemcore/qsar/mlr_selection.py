"""
MLR (Multiple Linear Regression) feature selection and evaluation utilities.

Implemented selection strategies:
- forward selection
- backward elimination
- Monte-Carlo (random subset search)
- Genetic algorithm (binary GA)

The typical workflow is:
1) impute missing values
2) low-variance filtering
3) high-correlation filtering
4) choose a selection strategy and optimize a subset for a chosen criterion
5) fit final LinearRegression
6) compute metrics + QSAR applicability domain (Williams/leverage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from sklearn.neighbors import NearestNeighbors

from scipy import stats


# -----------------------------------------------------------------------------
# Basic preprocessing
# -----------------------------------------------------------------------------

def impute_missing(X: NDArray[np.float64], strategy: str = "mean") -> NDArray[np.float64]:
    """Impute NaNs in X (2D) using sklearn SimpleImputer."""
    imp = SimpleImputer(strategy=strategy)
    return imp.fit_transform(X).astype(float, copy=False)


def filter_low_variance(
    X: NDArray[np.float64],
    feature_names: Sequence[str],
    threshold: float = 1e-12,
) -> Tuple[NDArray[np.float64], List[str], NDArray[np.int64]]:
    """
    Remove features with variance <= threshold.
    Returns (X_filtered, names_filtered, kept_indices_in_original).
    """
    if X.size == 0:
        return X, list(feature_names), np.arange(0, X.shape[1], dtype=int)

    var = np.nanvar(X, axis=0)
    keep = np.where(var > threshold)[0]
    return X[:, keep], [feature_names[i] for i in keep], keep.astype(int)


def filter_high_correlation(
    X: NDArray[np.float64],
    feature_names: Sequence[str],
    threshold: float = 0.95,
    prefer: str = "variance",
) -> Tuple[NDArray[np.float64], List[str], NDArray[np.int64]]:
    """
    Remove highly correlated features using a greedy procedure.

    For each pair with |corr| >= threshold, drop one feature.
    If prefer='variance', keep the higher-variance feature (robust default).
    Returns (X_filtered, names_filtered, kept_indices_in_original).
    """
    n, p = X.shape
    if p <= 1:
        return X, list(feature_names), np.arange(p, dtype=int)

    # correlation may have NaNs if a column is constant; handle by nan_to_num
    corr = np.corrcoef(np.nan_to_num(X, nan=0.0), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    var = np.nanvar(X, axis=0)
    keep = np.ones(p, dtype=bool)

    # order features by decreasing "priority" (variance) so we prefer to keep stable ones
    if prefer == "variance":
        order = np.argsort(-var)
    else:
        order = np.arange(p)

    for i in order:
        if not keep[i]:
            continue
        # find other features j still kept and correlated with i
        js = np.where((keep) & (np.arange(p) != i) & (np.abs(corr[i]) >= threshold))[0]
        for j in js:
            if not keep[j]:
                continue
            # decide which to drop
            if prefer == "variance":
                drop = j if var[i] >= var[j] else i
            else:
                drop = j  # default: drop the later one
            if drop == i:
                keep[i] = False
                break
            keep[drop] = False

    kept_idx = np.where(keep)[0].astype(int)
    return X[:, kept_idx], [feature_names[i] for i in kept_idx], kept_idx


# -----------------------------------------------------------------------------
# Metrics / criterion utilities
# -----------------------------------------------------------------------------

def rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


def concordance_correlation_coefficient(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Lin's concordance correlation coefficient (CCC)."""
    y_true = y_true.astype(float, copy=False)
    y_pred = y_pred.astype(float, copy=False)
    mu_x = float(np.mean(y_true))
    mu_y = float(np.mean(y_pred))
    var_x = float(np.var(y_true, ddof=0))
    var_y = float(np.var(y_pred, ddof=0))
    cov_xy = float(np.mean((y_true - mu_x) * (y_pred - mu_y)))
    denom = var_x + var_y + (mu_x - mu_y) ** 2
    if denom <= 0:
        return float("nan")
    return float(2.0 * cov_xy / denom)


def q2_f_metrics(
    y_train: NDArray[np.float64],
    y_test: NDArray[np.float64],
    y_pred_test: NDArray[np.float64],
) -> Dict[str, float]:
    """Compute Q²F1, Q²F2, Q²F3 external validation metrics."""
    y_train = y_train.astype(float, copy=False)
    y_test = y_test.astype(float, copy=False)
    y_pred_test = y_pred_test.astype(float, copy=False)

    press = float(np.sum((y_pred_test - y_test) ** 2))
    y_train_mean = float(np.mean(y_train))
    y_test_mean = float(np.mean(y_test))
    sst_train_mean = float(np.sum((y_test - y_train_mean) ** 2))
    sst_test_mean = float(np.sum((y_test - y_test_mean) ** 2))

    q2f1 = 1.0 - press / max(sst_train_mean, 1e-300)
    q2f2 = 1.0 - press / max(sst_test_mean, 1e-300)

    n_test = max(int(len(y_test)), 1)
    n_train = max(int(len(y_train)), 1)
    sst_train = float(np.sum((y_train - y_train_mean) ** 2))
    q2f3 = 1.0 - (press / n_test) / max((sst_train / n_train), 1e-300)

    return {"q2f1": float(q2f1), "q2f2": float(q2f2), "q2f3": float(q2f3)}


def _r0_squared_true_vs_pred(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """R0² for regression through origin: y_true ~ k * y_pred."""
    y_true = y_true.astype(float, copy=False)
    y_pred = y_pred.astype(float, copy=False)
    denom = float(np.sum(y_pred ** 2))
    if denom <= 0:
        return float("nan")
    k = float(np.sum(y_true * y_pred) / denom)
    y_fit = k * y_pred
    sse = float(np.sum((y_true - y_fit) ** 2))
    sst = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return float(1.0 - sse / max(sst, 1e-300))


def _r0_squared_pred_vs_true(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """R0'² for regression through origin: y_pred ~ k' * y_true."""
    y_true = y_true.astype(float, copy=False)
    y_pred = y_pred.astype(float, copy=False)
    denom = float(np.sum(y_true ** 2))
    if denom <= 0:
        return float("nan")
    k = float(np.sum(y_true * y_pred) / denom)
    y_fit = k * y_true
    sse = float(np.sum((y_pred - y_fit) ** 2))
    sst = float(np.sum((y_pred - float(np.mean(y_pred))) ** 2))
    return float(1.0 - sse / max(sst, 1e-300))


def rm2_metrics(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> Dict[str, float]:
    """Roy's r_m² metrics (r_m², r'_m², average, delta) and RTO slopes."""
    r2 = float(r2_score(y_true, y_pred))
    r0_2 = _r0_squared_true_vs_pred(y_true, y_pred)
    r0p_2 = _r0_squared_pred_vs_true(y_true, y_pred)

    rm2 = r2 * (1.0 - np.sqrt(max(r2 - r0_2, 0.0)))
    rm2p = r2 * (1.0 - np.sqrt(max(r2 - r0p_2, 0.0)))
    avg = 0.5 * (rm2 + rm2p)
    delta = abs(rm2 - rm2p)

    # slopes through origin
    k = float(np.sum(y_true * y_pred) / max(np.sum(y_pred ** 2), 1e-300))
    kprime = float(np.sum(y_true * y_pred) / max(np.sum(y_true ** 2), 1e-300))

    return {
        "r2": float(r2),
        "r0_2": float(r0_2),
        "r0p_2": float(r0p_2),
        "rm2": float(rm2),
        "rm2_prime": float(rm2p),
        "rm2_avg": float(avg),
        "rm2_delta": float(delta),
        "k": float(k),
        "k_prime": float(kprime),
    }


def external_validation_metrics(
    y_train: NDArray[np.float64],
    y_test: NDArray[np.float64],
    y_pred_test: NDArray[np.float64],
) -> Dict[str, float]:
    """Convenience bundle of common external-validation metrics for QSAR."""
    base = regression_metrics(y_test, y_pred_test)
    base["ccc"] = concordance_correlation_coefficient(y_test, y_pred_test)
    base.update(q2_f_metrics(y_train, y_test, y_pred_test))
    base.update({f"ext_{k}": v for k, v in rm2_metrics(y_test, y_pred_test).items()})
    return base


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R² for n samples and p predictors (excluding intercept)."""
    if n <= p + 1:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def aic_bic_from_rss(rss: float, n: int, k: int) -> Tuple[float, float]:
    """
    AIC and BIC from RSS under Gaussian errors.
    k is number of parameters incl. intercept.
    """
    if n <= 0:
        return float("nan"), float("nan")
    rss = max(float(rss), 1e-300)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + np.log(n) * k
    return float(aic), float(bic)


@dataclass(frozen=True)
class Criterion:
    """
    Criterion for subset selection.
    direction: +1 means maximize; -1 means minimize
    """
    name: str
    direction: int  # +1 maximize, -1 minimize


CRITERIA: Dict[str, Criterion] = {
    "cv_r2": Criterion("cv_r2", +1),
    "cv_rmse": Criterion("cv_rmse", -1),
    "adj_r2": Criterion("adj_r2", +1),
    "aic": Criterion("aic", -1),
    "bic": Criterion("bic", -1),
    "train_r2": Criterion("train_r2", +1),
}


def _cv_scores(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    cv: int,
    random_state: int,
) -> Tuple[float, float]:
    """Return (mean_cv_r2, mean_cv_rmse) using KFold CV on LinearRegression."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    r2s: List[float] = []
    rmses: List[float] = []
    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        model = LinearRegression()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        r2s.append(float(r2_score(yte, pred)))
        rmses.append(rmse(yte, pred))
    return float(np.mean(r2s)), float(np.mean(rmses))


def score_subset(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    subset: NDArray[np.int64],
    criterion: str = "cv_r2",
    cv: int = 5,
    random_state: int = 0,
) -> float:
    """
    Compute score for a subset of columns of X.
    Higher is better for maximize criteria; lower is better for minimize criteria.
    """
    if subset.size == 0:
        return float("-inf") if CRITERIA[criterion].direction == +1 else float("inf")

    Xs = X[:, subset]

    # guard: need n > p+1 for stable statistics
    n, p = Xs.shape
    if n <= p + 1:
        # penalize too many features
        return float("-inf") if CRITERIA[criterion].direction == +1 else float("inf")

    if criterion == "cv_r2" or criterion == "cv_rmse":
        cv_r2, cv_rmse = _cv_scores(Xs, y, cv=cv, random_state=random_state)
        return cv_r2 if criterion == "cv_r2" else cv_rmse

    # train-based criteria
    model = LinearRegression()
    model.fit(Xs, y)
    pred = model.predict(Xs)
    r2 = float(r2_score(y, pred))

    if criterion == "train_r2":
        return r2

    if criterion == "adj_r2":
        return adjusted_r2(r2, n=n, p=p)

    # AIC/BIC
    resid = y - pred
    rss = float(np.sum(resid * resid))
    aic, bic = aic_bic_from_rss(rss=rss, n=n, k=p + 1)
    return aic if criterion == "aic" else bic


def is_better(new_score: float, old_score: float, criterion: str) -> bool:
    """Compare scores given criterion direction."""
    direction = CRITERIA[criterion].direction
    return (new_score - old_score) * direction > 1e-12


# -----------------------------------------------------------------------------
# Applicability Domain (Williams plot)
# -----------------------------------------------------------------------------

def hat_diagonal(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute leverage (hat diagonal) for design matrix X (n x p), excluding intercept.
    Internally adds intercept column.
    """
    n, p = X.shape
    X1 = np.column_stack([np.ones(n), X])
    # h = diag( X (X^T X)^-1 X^T )
    xtx = X1.T @ X1
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    h = np.einsum("ij,jk,ik->i", X1, xtx_inv, X1)
    return h.astype(float)


def standardized_residuals(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    leverage: NDArray[np.float64],
    p: int,
) -> NDArray[np.float64]:
    """
    Standardized residuals for Williams plot.
    p = number of predictors (excluding intercept).
    """
    resid = y_true - y_pred
    n = len(y_true)
    dof = max(n - p - 1, 1)
    s = np.sqrt(np.sum(resid * resid) / dof)
    denom = s * np.sqrt(np.maximum(1.0 - leverage, 1e-12))
    return (resid / np.maximum(denom, 1e-12)).astype(float)


def leverage_threshold(n: int, p: int) -> float:
    """h* = 3(p+1)/n"""
    if n <= 0:
        return float("nan")
    return float(3.0 * (p + 1) / n)


@dataclass
class ADConfig:
    """Applicability Domain configuration."""

    use_williams: bool = True
    use_knn: bool = False
    use_mahalanobis: bool = False
    combine_mode: str = "and"  # and|or

    # kNN distance AD
    knn_k: int = 5
    knn_quantile: float = 0.95

    # Mahalanobis distance AD
    maha_alpha: float = 0.95
    maha_use_chi2: bool = True


def _standardize_fit(X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = np.mean(X, axis=0).astype(float)
    std = np.std(X, axis=0, ddof=0).astype(float)
    std = np.where(std > 1e-12, std, 1.0)
    return mean, std


def _standardize_apply(
    X: NDArray[np.float64],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
) -> NDArray[np.float64]:
    return ((X - mean) / std).astype(float)


def _knn_mean_distance(
    nn: NearestNeighbors,
    X_query: NDArray[np.float64],
    k: int,
) -> NDArray[np.float64]:
    """Mean distance to k nearest neighbors (excludes self if detected)."""
    k = max(int(k), 1)
    n_neighbors = min(k + 1, nn.n_samples_fit_)
    dists, _ = nn.kneighbors(X_query, n_neighbors=n_neighbors, return_distance=True)
    # If first neighbor is self (dist ~ 0), drop it.
    if dists.shape[1] > 1 and np.all(dists[:, 0] <= 1e-12):
        d_use = dists[:, 1 : min(1 + k, dists.shape[1])]
    else:
        d_use = dists[:, : min(k, dists.shape[1])]
    return np.mean(d_use, axis=1).astype(float)


def _mahalanobis_d2(
    X: NDArray[np.float64],
    mean: NDArray[np.float64],
    cov_inv: NDArray[np.float64],
) -> NDArray[np.float64]:
    d = X - mean
    return np.einsum("ij,jk,ik->i", d, cov_inv, d).astype(float)


# -----------------------------------------------------------------------------
# Coefficient statistics (optional but useful for reports)
# -----------------------------------------------------------------------------

def coefficient_statistics(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    model: LinearRegression,
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute standard errors, t-stats and p-values for coefficients (including intercept).
    Uses classical OLS assumptions.
    """
    n, p = X.shape
    X1 = np.column_stack([np.ones(n), X])
    beta = np.concatenate([[model.intercept_], model.coef_]).astype(float)

    y_pred = model.predict(X)
    resid = y - y_pred
    dof = max(n - (p + 1), 1)
    sigma2 = float((resid @ resid) / dof)

    xtx = X1.T @ X1
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)

    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 1e-300)).astype(float)
    tvals = (beta / np.maximum(se, 1e-300)).astype(float)
    pvals = (2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df=dof))).astype(float)

    return {"beta": beta, "se": se, "t": tvals, "p": pvals}


def vif(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Variance Inflation Factor per feature.
    VIF_j = 1 / (1 - R²_j), where R²_j is from regressing feature j on all others.
    """
    n, p = X.shape
    if p == 0:
        return np.array([], dtype=float)
    vifs = np.zeros(p, dtype=float)
    for j in range(p):
        yj = X[:, j]
        Xo = np.delete(X, j, axis=1)
        if Xo.shape[1] == 0:
            vifs[j] = 1.0
            continue
        reg = LinearRegression()
        reg.fit(Xo, yj)
        pred = reg.predict(Xo)
        r2 = float(r2_score(yj, pred))
        vifs[j] = 1.0 / max(1.0 - r2, 1e-12)
    return vifs


# -----------------------------------------------------------------------------
# Selection algorithms
# -----------------------------------------------------------------------------

@dataclass
class SelectionResult:
    selected_idx: NDArray[np.int64]
    selected_names: List[str]
    score: float
    history: List[Tuple[str, float, List[str]]]  # (step, score, names)


@dataclass
class SelectionConfig:
    method: str = "forward"  # forward|backward|montecarlo|genetic
    criterion: str = "cv_r2"
    cv_folds: int = 5
    random_state: int = 0
    max_features: int = 0  # 0 means no explicit max
    min_features: int = 1

    # Monte-Carlo
    mc_iterations: int = 2000

    # Genetic algorithm
    ga_population: int = 80
    ga_generations: int = 60
    ga_crossover: float = 0.7
    ga_mutation: float = 0.02
    ga_tournament: int = 3
    ga_elite: int = 2


def _apply_feature_constraints(mask: NDArray[np.bool_], min_features: int, max_features: int) -> NDArray[np.bool_]:
    """Ensure mask satisfies min/max by random adjust (used in GA)."""
    rng = np.random.default_rng()
    m = mask.size
    k = int(mask.sum())
    if max_features > 0 and k > max_features:
        on = np.where(mask)[0]
        drop = rng.choice(on, size=(k - max_features), replace=False)
        mask[drop] = False
    k = int(mask.sum())
    if k < min_features:
        off = np.where(~mask)[0]
        if off.size > 0:
            add = rng.choice(off, size=min(min_features - k, off.size), replace=False)
            mask[add] = True
    return mask


def forward_selection(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: Sequence[str],
    cfg: SelectionConfig,
) -> SelectionResult:
    p = X.shape[1]
    remaining = list(range(p))
    selected: List[int] = []
    history: List[Tuple[str, float, List[str]]] = []

    best_score = float("-inf") if CRITERIA[cfg.criterion].direction == +1 else float("inf")

    max_feat = cfg.max_features if cfg.max_features > 0 else p

    # greedy forward
    while remaining and len(selected) < max_feat:
        best_j = None
        best_local = best_score

        for j in remaining:
            cand = np.array(selected + [j], dtype=int)
            s = score_subset(X, y, cand, criterion=cfg.criterion, cv=cfg.cv_folds, random_state=cfg.random_state)
            if best_j is None or is_better(s, best_local, cfg.criterion):
                best_j = j
                best_local = s

        # stop if no improvement
        if best_j is None or not is_better(best_local, best_score, cfg.criterion):
            break

        selected.append(best_j)
        remaining.remove(best_j)
        best_score = best_local
        history.append((f"add {feature_names[best_j]}", best_score, [feature_names[i] for i in selected]))

    # enforce min_features if requested
    if len(selected) < cfg.min_features:
        # fill best remaining by one-step look-ahead
        while remaining and len(selected) < cfg.min_features:
            best_j = None
            best_local = best_score
            for j in remaining:
                cand = np.array(selected + [j], dtype=int)
                s = score_subset(X, y, cand, criterion=cfg.criterion, cv=cfg.cv_folds, random_state=cfg.random_state)
                if best_j is None or is_better(s, best_local, cfg.criterion):
                    best_j = j
                    best_local = s
            if best_j is None:
                break
            selected.append(best_j)
            remaining.remove(best_j)
            best_score = best_local
            history.append((f"force-add {feature_names[best_j]}", best_score, [feature_names[i] for i in selected]))

    sel_idx = np.array(sorted(selected), dtype=int)
    return SelectionResult(sel_idx, [feature_names[i] for i in sel_idx], float(best_score), history)


def backward_elimination(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: Sequence[str],
    cfg: SelectionConfig,
) -> SelectionResult:
    p = X.shape[1]
    selected = list(range(p))
    history: List[Tuple[str, float, List[str]]] = []

    max_feat = cfg.max_features if cfg.max_features > 0 else p
    if len(selected) > max_feat:
        # initial trim using variance (keep high variance)
        var = np.nanvar(X, axis=0)
        order = list(np.argsort(-var))
        selected = order[:max_feat]

    best_score = score_subset(
        X, y, np.array(selected, dtype=int),
        criterion=cfg.criterion, cv=cfg.cv_folds, random_state=cfg.random_state
    )
    history.append(("start", float(best_score), [feature_names[i] for i in selected]))

    while len(selected) > max(cfg.min_features, 1):
        best_drop = None
        best_local = best_score

        for j in list(selected):
            cand = np.array([i for i in selected if i != j], dtype=int)
            s = score_subset(X, y, cand, criterion=cfg.criterion, cv=cfg.cv_folds, random_state=cfg.random_state)
            if best_drop is None or is_better(s, best_local, cfg.criterion):
                best_drop = j
                best_local = s

        if best_drop is None or not is_better(best_local, best_score, cfg.criterion):
            break

        selected.remove(best_drop)
        best_score = best_local
        history.append((f"drop {feature_names[best_drop]}", best_score, [feature_names[i] for i in selected]))

    sel_idx = np.array(sorted(selected), dtype=int)
    return SelectionResult(sel_idx, [feature_names[i] for i in sel_idx], float(best_score), history)


def monte_carlo_selection(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: Sequence[str],
    cfg: SelectionConfig,
) -> SelectionResult:
    rng = np.random.default_rng(cfg.random_state)
    p = X.shape[1]
    max_feat = cfg.max_features if cfg.max_features > 0 else min(p, max(cfg.min_features, 20))
    min_feat = max(cfg.min_features, 1)

    best_subset = None
    best_score = float("-inf") if CRITERIA[cfg.criterion].direction == +1 else float("inf")
    history: List[Tuple[str, float, List[str]]] = []

    for it in range(int(cfg.mc_iterations)):
        k = int(rng.integers(min_feat, max_feat + 1))
        subset = np.sort(rng.choice(p, size=k, replace=False)).astype(int)
        s = score_subset(X, y, subset, criterion=cfg.criterion, cv=cfg.cv_folds, random_state=cfg.random_state)
        if best_subset is None or is_better(s, best_score, cfg.criterion):
            best_subset = subset
            best_score = s
            if it % 25 == 0:
                history.append((f"iter {it}", float(best_score), [feature_names[i] for i in best_subset]))

    if best_subset is None:
        best_subset = np.array([], dtype=int)

    return SelectionResult(best_subset, [feature_names[i] for i in best_subset], float(best_score), history)


def _tournament_select(fitness: NDArray[np.float64], k: int, rng: np.random.Generator) -> int:
    idx = rng.choice(len(fitness), size=k, replace=False)
    # higher fitness is always better
    return int(idx[np.argmax(fitness[idx])])


def genetic_algorithm_selection(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: Sequence[str],
    cfg: SelectionConfig,
) -> SelectionResult:
    """
    Simple binary GA with tournament selection and one-point crossover.
    Fitness is always maximized (we transform criterion accordingly).
    """
    rng = np.random.default_rng(cfg.random_state)
    p = X.shape[1]

    max_feat = cfg.max_features if cfg.max_features > 0 else min(p, max(cfg.min_features, 20))
    min_feat = max(cfg.min_features, 1)

    pop_size = max(int(cfg.ga_population), 10)
    gens = max(int(cfg.ga_generations), 1)

    # transform score to fitness: maximize in all cases
    direction = CRITERIA[cfg.criterion].direction

    def _fitness(mask: NDArray[np.bool_]) -> float:
        mask = mask.copy()
        mask = _apply_feature_constraints(mask, min_features=min_feat, max_features=max_feat)
        subset = np.where(mask)[0].astype(int)
        s = score_subset(X, y, subset, criterion=cfg.criterion, cv=cfg.cv_folds, random_state=cfg.random_state)
        if np.isnan(s):
            s = float("-inf") if direction == +1 else float("inf")
        return float(s * direction)

    # init population (biased toward smaller subsets)
    pop = rng.random((pop_size, p)) < 0.15
    for i in range(pop_size):
        pop[i] = _apply_feature_constraints(pop[i], min_features=min_feat, max_features=max_feat)

    fit = np.array([_fitness(ind) for ind in pop], dtype=float)

    history: List[Tuple[str, float, List[str]]] = []
    best_i = int(np.argmax(fit))
    best_mask = pop[best_i].copy()
    best_fit = float(fit[best_i])
    best_score = float(best_fit / direction)

    history.append(("init", best_score, [feature_names[i] for i in np.where(best_mask)[0]]))

    for g in range(gens):
        # elitism
        elite_n = max(int(cfg.ga_elite), 0)
        elite_idx = np.argsort(-fit)[:elite_n]
        new_pop = [pop[i].copy() for i in elite_idx]

        # create offspring
        while len(new_pop) < pop_size:
            p1 = pop[_tournament_select(fit, cfg.ga_tournament, rng)]
            p2 = pop[_tournament_select(fit, cfg.ga_tournament, rng)]
            c1, c2 = p1.copy(), p2.copy()

            # crossover
            if rng.random() < float(cfg.ga_crossover) and p > 1:
                cut = int(rng.integers(1, p))
                c1[:cut], c2[:cut] = p2[:cut], p1[:cut]

            # mutation
            mut = float(cfg.ga_mutation)
            if mut > 0:
                m1 = rng.random(p) < mut
                m2 = rng.random(p) < mut
                c1[m1] = ~c1[m1]
                c2[m2] = ~c2[m2]

            c1 = _apply_feature_constraints(c1, min_features=min_feat, max_features=max_feat)
            c2 = _apply_feature_constraints(c2, min_features=min_feat, max_features=max_feat)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop, dtype=bool)
        fit = np.array([_fitness(ind) for ind in pop], dtype=float)

        best_i = int(np.argmax(fit))
        if fit[best_i] > best_fit + 1e-12:
            best_fit = float(fit[best_i])
            best_mask = pop[best_i].copy()
            best_score = float(best_fit / direction)
            history.append((f"gen {g+1}", best_score, [feature_names[i] for i in np.where(best_mask)[0]]))

    best_idx = np.where(best_mask)[0].astype(int)
    return SelectionResult(best_idx, [feature_names[i] for i in best_idx], float(best_score), history)


def select_features(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: Sequence[str],
    cfg: SelectionConfig,
) -> SelectionResult:
    method = cfg.method.lower().strip()
    if method == "forward":
        return forward_selection(X, y, feature_names, cfg)
    if method == "backward":
        return backward_elimination(X, y, feature_names, cfg)
    if method in {"mc", "montecarlo", "monte-carlo"}:
        return monte_carlo_selection(X, y, feature_names, cfg)
    if method in {"ga", "genetic", "genetic-algorithm", "genetic_algorithm"}:
        return genetic_algorithm_selection(X, y, feature_names, cfg)
    raise ValueError(f"Unknown selection method: {cfg.method!r}")


# -----------------------------------------------------------------------------
# High-level helper: build model + per-row diagnostics
# -----------------------------------------------------------------------------

@dataclass
class MLRFitResult:
    model: LinearRegression
    selected_names: List[str]
    selected_idx: NDArray[np.int64]
    metrics_train: Dict[str, float]
    cv_metrics: Dict[str, float]
    coef_stats: Dict[str, NDArray[np.float64]]
    vifs: NDArray[np.float64]
    leverage_train: NDArray[np.float64]
    stdres_train: NDArray[np.float64]
    h_star: float
    y_pred_train: NDArray[np.float64]

    # Applicability Domain (distance-based)
    ad_cfg: ADConfig
    z_mean: NDArray[np.float64]
    z_std: NDArray[np.float64]
    knn_model: Optional[NearestNeighbors]
    knn_threshold: Optional[float]
    maha_mean: Optional[NDArray[np.float64]]
    maha_cov_inv: Optional[NDArray[np.float64]]
    maha_threshold: Optional[float]


def fit_mlr_with_selection(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: Sequence[str],
    cfg: SelectionConfig,
    ad_cfg: Optional[ADConfig] = None,
) -> Tuple[MLRFitResult, SelectionResult]:
    sel = select_features(X, y, feature_names, cfg)
    Xs = X[:, sel.selected_idx]

    model = LinearRegression()
    model.fit(Xs, y)

    y_pred = model.predict(Xs)
    mtrain = regression_metrics(y, y_pred)

    # CV metrics on selected space
    cv_r2, cv_rmse = _cv_scores(Xs, y, cv=cfg.cv_folds, random_state=cfg.random_state)
    cv_metrics = {"q2": float(cv_r2), "rmse_cv": float(cv_rmse)}

    leverage = hat_diagonal(Xs)
    stdres = standardized_residuals(y, y_pred, leverage, p=Xs.shape[1])
    h_star = leverage_threshold(n=len(y), p=Xs.shape[1])

    # distance-based AD models are built in standardized selected space
    if ad_cfg is None:
        ad_cfg = ADConfig()
    z_mean, z_std = _standardize_fit(Xs)
    Xz = _standardize_apply(Xs, z_mean, z_std)

    knn_model: Optional[NearestNeighbors] = None
    knn_threshold: Optional[float] = None
    if ad_cfg.use_knn and len(Xz) >= 2:
        knn_model = NearestNeighbors(n_neighbors=min(int(ad_cfg.knn_k) + 1, len(Xz)))
        knn_model.fit(Xz)
        d_train = _knn_mean_distance(knn_model, Xz, k=int(ad_cfg.knn_k))
        knn_threshold = float(np.quantile(d_train, float(ad_cfg.knn_quantile)))

    maha_mean: Optional[NDArray[np.float64]] = None
    maha_cov_inv: Optional[NDArray[np.float64]] = None
    maha_threshold: Optional[float] = None
    if ad_cfg.use_mahalanobis and len(Xz) >= 3:
        maha_mean = np.mean(Xz, axis=0).astype(float)
        cov = np.cov(Xz, rowvar=False)
        try:
            maha_cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            maha_cov_inv = np.linalg.pinv(cov)

        d2_train = _mahalanobis_d2(Xz, maha_mean, maha_cov_inv)
        if ad_cfg.maha_use_chi2:
            df = max(int(Xz.shape[1]), 1)
            maha_threshold = float(stats.chi2.ppf(float(ad_cfg.maha_alpha), df=df))
        else:
            maha_threshold = float(np.quantile(d2_train, float(ad_cfg.maha_alpha)))

    coef_stats = coefficient_statistics(Xs, y, model)
    vifs = vif(Xs)

    return (
        MLRFitResult(
            model=model,
            selected_names=sel.selected_names,
            selected_idx=sel.selected_idx,
            metrics_train=mtrain,
            cv_metrics=cv_metrics,
            coef_stats=coef_stats,
            vifs=vifs,
            leverage_train=leverage,
            stdres_train=stdres,
            h_star=h_star,
            y_pred_train=y_pred,

            ad_cfg=ad_cfg,
            z_mean=z_mean,
            z_std=z_std,
            knn_model=knn_model,
            knn_threshold=knn_threshold,
            maha_mean=maha_mean,
            maha_cov_inv=maha_cov_inv,
            maha_threshold=maha_threshold,
        ),
        sel,
    )


def predict_with_ad(
    fit: MLRFitResult,
    X: NDArray[np.float64],
    y_true: Optional[NDArray[np.float64]] = None,
) -> Dict[str, Any]:
    """
    Predict and compute applicability domain diagnostics.

    The returned key `in_ad` is a combined flag using the enabled AD methods
    (Williams/leverage, kNN distance, Mahalanobis distance). If `y_true` is
    provided, |standardized residual| <= 3 is also included.
    """
    Xs = X[:, fit.selected_idx]
    y_pred = fit.model.predict(Xs).astype(float)

    lev = hat_diagonal(Xs)

    # standardize selected space (distance-based AD)
    Xz = _standardize_apply(Xs, fit.z_mean, fit.z_std)

    in_williams = (lev <= fit.h_star) if fit.ad_cfg.use_williams else np.ones(len(lev), dtype=bool)

    knn_dist = None
    in_knn = np.ones(len(lev), dtype=bool)
    if fit.ad_cfg.use_knn and fit.knn_model is not None and fit.knn_threshold is not None:
        knn_dist = _knn_mean_distance(fit.knn_model, Xz, k=int(fit.ad_cfg.knn_k))
        in_knn = knn_dist <= float(fit.knn_threshold)

    maha_d2 = None
    in_maha = np.ones(len(lev), dtype=bool)
    if fit.ad_cfg.use_mahalanobis and fit.maha_cov_inv is not None and fit.maha_threshold is not None and fit.maha_mean is not None:
        maha_d2 = _mahalanobis_d2(Xz, fit.maha_mean, fit.maha_cov_inv)
        in_maha = maha_d2 <= float(fit.maha_threshold)

    # combine AD methods
    combine = (fit.ad_cfg.combine_mode or "and").lower().strip()
    if combine == "or":
        in_ad_base = in_williams | in_knn | in_maha
    else:
        in_ad_base = in_williams & in_knn & in_maha

    if y_true is not None:
        stdres = standardized_residuals(y_true.astype(float), y_pred, lev, p=Xs.shape[1])
        in_ad = in_ad_base & (np.abs(stdres) <= 3.0)
    else:
        stdres = None
        in_ad = in_ad_base

    return {
        "y_pred": y_pred,
        "leverage": lev.astype(float),
        "std_resid": None if stdres is None else stdres.astype(float),
        "h_star": float(fit.h_star),
        "in_ad": in_ad.astype(bool),
        "in_ad_williams": in_williams.astype(bool),
        "knn_dist": None if knn_dist is None else knn_dist.astype(float),
        "knn_threshold": None if fit.knn_threshold is None else float(fit.knn_threshold),
        "in_ad_knn": in_knn.astype(bool),
        "maha_d2": None if maha_d2 is None else maha_d2.astype(float),
        "maha_threshold": None if fit.maha_threshold is None else float(fit.maha_threshold),
        "in_ad_maha": in_maha.astype(bool),
    }


def permutation_test_cv_q2(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    subset: NDArray[np.int64],
    n_permutations: int = 100,
    cv: int = 5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Y-randomization / permutation test using CV R² (Q²) as statistic.
    Returns observed Q², permuted distribution and p-value.
    """
    rng = np.random.default_rng(random_state)
    Xs = X[:, subset]
    obs, _ = _cv_scores(Xs, y, cv=cv, random_state=random_state)
    perm = []
    for _ in range(int(n_permutations)):
        yp = rng.permutation(y)
        q2, _ = _cv_scores(Xs, yp, cv=cv, random_state=random_state)
        perm.append(float(q2))
    perm = np.array(perm, dtype=float)
    # p-value: how often permuted >= observed
    pval = float((np.sum(perm >= obs) + 1) / (len(perm) + 1))
    return {"q2_observed": float(obs), "q2_perm": perm, "p_value": pval}
