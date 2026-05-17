from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
from Orange.data import Table

from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table


DEFAULT_QSAR_TARGET_COLUMN = "pActivity"

TARGET_COLUMN_CANDIDATES: tuple[str, ...] = (
    "pactivity",
    "activity",
    "target",
    "label",
    "y",
    "boiling_point",
    "bp",
    "tbp",
    "t_bp",
    "melting_point",
    "mp",
    "tmp",
    "t_mp",
    "flash_point",
    "fp",
    "logs",
    "solubility",
    "log_solubility",
    "logp",
    "log_p",
    "logd",
    "log_d",
    "viscosity",
    "density",
    "pka",
    "pkb",
    "ic50",
    "pic50",
    "ec50",
    "pec50",
    "ki",
    "pki",
    "clearance",
    "half_life",
    "mw",
    "molwt",
)


def normalize_target_label(value: Any, *, fallback: str = DEFAULT_QSAR_TARGET_COLUMN) -> str:
    text = str(value or "").strip()
    return text or fallback


def safe_prediction_identifier(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "value"
    out = []
    for ch in raw:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "value"


def prediction_column_name_for_target(target_label: Any) -> str:
    return f"predicted_{safe_prediction_identifier(normalize_target_label(target_label, fallback='value'))}"


def qsar_ready_class_column() -> str:
    return DEFAULT_QSAR_TARGET_COLUMN


def build_qsar_ready_table(records, *, name: str = "QSAR Ready Data"):
    return records_to_orange_table(
        records,
        class_column=qsar_ready_class_column(),
        name=name,
        numeric_as_attributes=True,
    )


def _norm_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _is_continuous_var(var: Any) -> bool:
    return bool(getattr(var, "is_continuous", False))


def _column_numeric_like(data: Table, var: Any) -> bool:
    if _is_continuous_var(var):
        return True
    try:
        values = data.get_column(var)
    except Exception:
        return False
    saw = False
    for value in values:
        try:
            val = float(value)
        except Exception:
            return False
        if np.isfinite(val):
            saw = True
    return saw


def _candidate_name_match(var_name: str, candidates: Sequence[str]) -> bool:
    low = _norm_name(var_name)
    return any(_norm_name(cand) == low for cand in candidates)


def preferred_target_name_from_table(
    data: Table | None,
    *,
    candidates: Sequence[str] = TARGET_COLUMN_CANDIDATES,
) -> str:
    if data is None:
        return ""
    class_var = getattr(data.domain, "class_var", None)
    if class_var is not None and getattr(class_var, "name", None) and _is_continuous_var(class_var):
        return str(class_var.name)

    ordered_vars: list[Any] = (
        list(data.domain.class_vars)
        + list(data.domain.attributes)
        + list(data.domain.metas)
    )
    normalized_vars = {
        _norm_name(getattr(var, "name", "")): var
        for var in ordered_vars
        if getattr(var, "name", None)
    }
    for candidate in candidates:
        var = normalized_vars.get(_norm_name(candidate))
        if var is not None and _column_numeric_like(data, var):
            return str(var.name)
    return ""


def infer_target_label_from_model(
    model: Any,
    *,
    configured: Any = "",
    fallback: str = DEFAULT_QSAR_TARGET_COLUMN,
) -> str:
    configured_text = str(configured or "").strip()
    if configured_text:
        return configured_text
    if hasattr(model, "target_label") and getattr(model, "target_label"):
        return normalize_target_label(getattr(model, "target_label"), fallback=fallback)
    if hasattr(model, "y_name") and getattr(model, "y_name"):
        return normalize_target_label(getattr(model, "y_name"), fallback=fallback)
    return fallback


__all__ = [
    "DEFAULT_QSAR_TARGET_COLUMN",
    "TARGET_COLUMN_CANDIDATES",
    "build_qsar_ready_table",
    "infer_target_label_from_model",
    "normalize_target_label",
    "prediction_column_name_for_target",
    "preferred_target_name_from_table",
    "qsar_ready_class_column",
    "safe_prediction_identifier",
]
