from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

from chem_inf_widgets.chemcore.qsar.mlr_selection import ADConfig, leverage_threshold


@dataclass(frozen=True)
class ApplicabilityDomainFit:
    feature_names: tuple[str, ...]
    ad_cfg: ADConfig
    imputer: SimpleImputer
    ref_xtx_inv: np.ndarray
    ref_feature_count: int
    ref_row_count: int
    h_star: float
    z_mean: np.ndarray
    z_std: np.ndarray
    knn_model: Optional[NearestNeighbors]
    knn_threshold: Optional[float]
    maha_mean: Optional[np.ndarray]
    maha_cov_inv: Optional[np.ndarray]
    maha_threshold: Optional[float]


@dataclass(frozen=True)
class ApplicabilityDomainPrediction:
    leverage: np.ndarray
    in_ad: np.ndarray
    in_ad_williams: np.ndarray
    knn_dist: Optional[np.ndarray]
    in_ad_knn: np.ndarray
    maha_d2: Optional[np.ndarray]
    in_ad_maha: np.ndarray
    h_star: float
    knn_threshold: Optional[float]
    maha_threshold: Optional[float]


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D numeric matrix.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Applicability Domain requires at least one row and one feature.")
    return arr


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0).astype(float)
    std = np.std(X, axis=0, ddof=0).astype(float)
    std = np.where(std > 1e-12, std, 1.0)
    return mean, std


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(float)


def _hat_matrix_inverse(X_ref: np.ndarray) -> np.ndarray:
    n = X_ref.shape[0]
    X1 = np.column_stack([np.ones(n), X_ref])
    xtx = X1.T @ X1
    try:
        return np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx)


def _external_leverage(X_query: np.ndarray, xtx_inv: np.ndarray) -> np.ndarray:
    n = X_query.shape[0]
    X1 = np.column_stack([np.ones(n), X_query])
    return np.einsum("ij,jk,ik->i", X1, xtx_inv, X1).astype(float)


def _knn_mean_distance(nn: NearestNeighbors, X_query: np.ndarray, k: int) -> np.ndarray:
    k = max(int(k), 1)
    n_neighbors = min(k + 1, nn.n_samples_fit_)
    dists, _ = nn.kneighbors(X_query, n_neighbors=n_neighbors, return_distance=True)
    if dists.shape[1] > 1 and np.all(dists[:, 0] <= 1e-12):
        d_use = dists[:, 1 : min(1 + k, dists.shape[1])]
    else:
        d_use = dists[:, : min(k, dists.shape[1])]
    return np.mean(d_use, axis=1).astype(float)


def _mahalanobis_d2(X: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    d = X - mean
    return np.einsum("ij,jk,ik->i", d, cov_inv, d).astype(float)


def fit_applicability_domain(
    X_reference: np.ndarray,
    feature_names: Sequence[str],
    *,
    ad_cfg: Optional[ADConfig] = None,
) -> ApplicabilityDomainFit:
    X_ref_raw = _as_2d_float(X_reference)
    if len(feature_names) != X_ref_raw.shape[1]:
        raise ValueError("Feature name count does not match the number of columns.")

    cfg = ad_cfg or ADConfig()
    imputer = SimpleImputer(strategy="median")
    X_ref = imputer.fit_transform(X_ref_raw).astype(float)

    z_mean, z_std = _standardize_fit(X_ref)
    X_ref_z = _standardize_apply(X_ref, z_mean, z_std)
    xtx_inv = _hat_matrix_inverse(X_ref)

    knn_model: Optional[NearestNeighbors] = None
    knn_threshold: Optional[float] = None
    if cfg.use_knn and len(X_ref_z) >= 2:
        knn_model = NearestNeighbors(n_neighbors=min(int(cfg.knn_k) + 1, len(X_ref_z)))
        knn_model.fit(X_ref_z)
        d_train = _knn_mean_distance(knn_model, X_ref_z, k=int(cfg.knn_k))
        knn_threshold = float(np.quantile(d_train, float(cfg.knn_quantile)))

    maha_mean: Optional[np.ndarray] = None
    maha_cov_inv: Optional[np.ndarray] = None
    maha_threshold: Optional[float] = None
    if cfg.use_mahalanobis and len(X_ref_z) >= 3:
        maha_mean = np.mean(X_ref_z, axis=0).astype(float)
        cov = np.atleast_2d(np.cov(X_ref_z, rowvar=False))
        try:
            maha_cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            maha_cov_inv = np.linalg.pinv(cov)

        d2_train = _mahalanobis_d2(X_ref_z, maha_mean, maha_cov_inv)
        if cfg.maha_use_chi2:
            df = max(int(X_ref_z.shape[1]), 1)
            maha_threshold = float(stats.chi2.ppf(float(cfg.maha_alpha), df=df))
        else:
            maha_threshold = float(np.quantile(d2_train, float(cfg.maha_alpha)))

    return ApplicabilityDomainFit(
        feature_names=tuple(feature_names),
        ad_cfg=cfg,
        imputer=imputer,
        ref_xtx_inv=xtx_inv,
        ref_feature_count=int(X_ref.shape[1]),
        ref_row_count=int(X_ref.shape[0]),
        h_star=leverage_threshold(n=len(X_ref), p=X_ref.shape[1]),
        z_mean=z_mean,
        z_std=z_std,
        knn_model=knn_model,
        knn_threshold=knn_threshold,
        maha_mean=maha_mean,
        maha_cov_inv=maha_cov_inv,
        maha_threshold=maha_threshold,
    )


def score_applicability_domain(
    fit: ApplicabilityDomainFit,
    X_query: np.ndarray,
) -> ApplicabilityDomainPrediction:
    X_query_raw = _as_2d_float(X_query)
    if X_query_raw.shape[1] != len(fit.feature_names):
        raise ValueError("Query feature count does not match the fitted Applicability Domain.")

    X_query_imp = fit.imputer.transform(X_query_raw).astype(float)
    leverage = _external_leverage(X_query_imp, fit.ref_xtx_inv)
    X_query_z = _standardize_apply(X_query_imp, fit.z_mean, fit.z_std)

    use_williams = bool(fit.ad_cfg.use_williams)
    in_williams = (leverage <= fit.h_star) if use_williams else np.ones(len(leverage), dtype=bool)

    knn_dist = None
    in_knn = np.ones(len(leverage), dtype=bool)
    if fit.ad_cfg.use_knn and fit.knn_model is not None and fit.knn_threshold is not None:
        knn_dist = _knn_mean_distance(fit.knn_model, X_query_z, k=int(fit.ad_cfg.knn_k))
        in_knn = knn_dist <= float(fit.knn_threshold)

    maha_d2 = None
    in_maha = np.ones(len(leverage), dtype=bool)
    if (
        fit.ad_cfg.use_mahalanobis
        and fit.maha_cov_inv is not None
        and fit.maha_mean is not None
        and fit.maha_threshold is not None
    ):
        maha_d2 = _mahalanobis_d2(X_query_z, fit.maha_mean, fit.maha_cov_inv)
        in_maha = maha_d2 <= float(fit.maha_threshold)

    enabled_flags = []
    if fit.ad_cfg.use_williams:
        enabled_flags.append(in_williams)
    if fit.ad_cfg.use_knn:
        enabled_flags.append(in_knn)
    if fit.ad_cfg.use_mahalanobis:
        enabled_flags.append(in_maha)

    if not enabled_flags:
        in_ad = np.ones(len(leverage), dtype=bool)
    elif (fit.ad_cfg.combine_mode or "and").strip().lower() == "or":
        in_ad = enabled_flags[0].copy()
        for flag in enabled_flags[1:]:
            in_ad |= flag
    else:
        in_ad = enabled_flags[0].copy()
        for flag in enabled_flags[1:]:
            in_ad &= flag

    return ApplicabilityDomainPrediction(
        leverage=leverage.astype(float),
        in_ad=in_ad.astype(bool),
        in_ad_williams=in_williams.astype(bool),
        knn_dist=None if knn_dist is None else knn_dist.astype(float),
        in_ad_knn=in_knn.astype(bool),
        maha_d2=None if maha_d2 is None else maha_d2.astype(float),
        in_ad_maha=in_maha.astype(bool),
        h_star=float(fit.h_star),
        knn_threshold=None if fit.knn_threshold is None else float(fit.knn_threshold),
        maha_threshold=None if fit.maha_threshold is None else float(fit.maha_threshold),
    )
