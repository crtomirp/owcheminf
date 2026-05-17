from __future__ import annotations

from typing import Final

import numpy as np


SELECTION_TOOL_OPTIONS: Final[tuple[str, str]] = ("Rectangle", "Lasso")


def display_model_name(model_key: str) -> str:
    return str(model_key or "").replace("_", " ").title()


def residual_reference_levels(residuals) -> dict[str, float]:
    residuals_arr = np.asarray(residuals, dtype=float)
    finite = residuals_arr[np.isfinite(residuals_arr)]
    if finite.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "minus_1std": 0.0,
            "plus_1std": 0.0,
            "minus_2std": 0.0,
            "plus_2std": 0.0,
        }
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return {
        "mean": mean,
        "std": std,
        "minus_1std": mean - std,
        "plus_1std": mean + std,
        "minus_2std": mean - (2.0 * std),
        "plus_2std": mean + (2.0 * std),
    }
