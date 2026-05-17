from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from sklearn.feature_selection import f_regression


def safe_f_regression(X, y, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Robust ``f_regression`` for descriptor matrices.

    RDKit/Mordred descriptor tables can contain constant, near-constant or
    numerically unstable columns. Scikit-learn can emit repeated RuntimeWarning
    messages from an internal sqrt calculation. For feature selection, those
    columns should simply receive a non-informative score instead of polluting
    the terminal or crashing model selection.
    """

    kwargs.setdefault("force_finite", True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        scores, pvalues = f_regression(X, y, *args, **kwargs)
    scores = np.asarray(scores, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    scores[~np.isfinite(scores)] = 0.0
    pvalues[~np.isfinite(pvalues)] = 1.0
    return scores, pvalues
