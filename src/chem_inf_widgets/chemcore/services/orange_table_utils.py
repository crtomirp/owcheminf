"""Shared Orange Table construction helpers for cheminformatics workflows.

The central convention used across the add-on is:

- structure identifiers/provenance fields -> metas
- measured/modeling numeric values and descriptors -> attributes
- selected QSAR endpoint (for example pActivity) -> class_var

Keeping this logic in one place prevents widget-specific Domain drift where
important numeric columns silently become metas and disappear from downstream
Orange learners.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table


_ALWAYS_META_NAMES = {
    "smiles",
    "input_smiles",
    "canonical_smiles",
    "standardized_smiles",
    "canonical_smiles_std",
    "mol_smiles",
    "structure",
    "name",
    "title",
    "compound",
    "compound_name",
    "compound_id",
    "molecule_id",
    "id",
    "identifier",
    "chembl_id",
    "inchi",
    "inchikey",
    "identity_key",
    "identity_status",
    "source",
    "source_name",
    "source_format",
    "row_id",
    "transform_log",
    "dropped_reason",
    "qc_flags",
    "import_source_format",
    "import_row_index",
    "import_delimiter",
    "import_warnings",
    "unit",
    "units",
    "standard_units",
    "activity_unit",
    "relation",
    "standard_relation",
    "endpoint",
    "standard_type",
    "activity_type",
    "conversion_status",
    "reject_reasons",
    "aggregation_method",
    "row_index",
    "activity_value",
    "pactivity_raw",
    "pactivity_min",
    "pactivity_max",
    "pactivity_std",
    "n_measurements",
    "duplicate_group",
    "curation_stage",
    "curation_status",
    "curation_ready_for_qsar",
    "curation_ready_for_docking",
    "curation_blockers",
    "curation_warnings",
    "curation_recommended_next_step",
    "curation_version",
}


def looks_like_meta_key(name: str) -> bool:
    """Return True for columns that should usually remain Orange metas."""
    low = str(name).strip().lower()
    return (
        low in _ALWAYS_META_NAMES
        or low.endswith("_id")
        or low.endswith("_ids")
        or low.endswith("_name")
        or low.endswith("_smiles")
        or low.startswith("import_")
    )


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "?", "nan", "None"}
    try:
        return bool(math.isnan(float(value)))
    except Exception:
        return False


def as_float_or_nan(value: Any) -> float:
    if is_missing(value):
        return float("nan")
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        return float(value)
    except Exception:
        return float("nan")


def column_is_numeric(values: Sequence[Any]) -> bool:
    """True when all present values can be represented as floats."""
    saw_value = False
    for value in values:
        if is_missing(value):
            continue
        try:
            float(str(value).strip().replace(",", "."))
        except Exception:
            return False
        saw_value = True
    return saw_value


def _unique_variable_name(existing: set[str], wanted: str) -> str:
    if wanted not in existing:
        existing.add(wanted)
        return wanted
    i = 2
    while f"{wanted}_{i}" in existing:
        i += 1
    out = f"{wanted}_{i}"
    existing.add(out)
    return out


def _as_2d_array(values: Any, *, n_rows: int, dtype: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] != n_rows:
        raise ValueError(f"Expected {n_rows} rows, got {arr.shape[0]}.")
    return arr


def safe_table_from_numpy(
    domain: Domain,
    *,
    X: Any | None = None,
    Y: Any | None = None,
    metas: Any | None = None,
    name: str = "Data",
) -> Table:
    """Create an Orange Table while normalising empty ``Y``/``metas`` arrays.

    Orange's overloaded ``Table.from_numpy`` is strict: an empty ``Y`` array
    such as shape ``(n, 0)`` can fail even when the domain has no class
    variables. This helper keeps widget outputs stable for modeling tables,
    descriptor tables, reports, and empty-but-valid filtered outputs.
    """
    n_rows = 0
    if X is not None:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n_rows = int(X_arr.shape[0])
    elif metas is not None:
        M_tmp = np.asarray(metas, dtype=object)
        n_rows = int(M_tmp.shape[0]) if M_tmp.ndim else 1
        X_arr = np.empty((n_rows, len(domain.attributes)), dtype=float)
    elif Y is not None:
        Y_tmp = np.asarray(Y, dtype=float)
        n_rows = int(Y_tmp.shape[0]) if Y_tmp.ndim else 1
        X_arr = np.empty((n_rows, len(domain.attributes)), dtype=float)
    else:
        X_arr = np.empty((0, len(domain.attributes)), dtype=float)

    if X_arr.ndim != 2:
        X_arr = X_arr.reshape(n_rows, -1)
    if len(domain.attributes) == 0 and X_arr.shape[1] != 0:
        X_arr = np.empty((n_rows, 0), dtype=float)

    if len(domain.class_vars) == 0:
        Y_arr = None
    elif Y is None:
        Y_arr = None
    else:
        raw_y = np.asarray(Y, dtype=float)
        if raw_y.size == 0 or (raw_y.ndim == 2 and raw_y.shape[1] == 0):
            Y_arr = None
        else:
            Y_arr = raw_y
            if len(domain.class_vars) > 1:
                Y_arr = _as_2d_array(Y_arr, n_rows=n_rows, dtype=float)
            elif np.asarray(Y_arr).ndim == 2 and np.asarray(Y_arr).shape[1] == 1:
                # Orange accepts 1D Y for a single class variable more reliably.
                Y_arr = np.asarray(Y_arr, dtype=float).reshape(-1)

    if len(domain.metas) == 0:
        M_arr = None
    elif metas is None:
        M_arr = np.empty((n_rows, len(domain.metas)), dtype=object)
        M_arr[:, :] = ""
    else:
        raw_m = np.asarray(metas, dtype=object)
        if raw_m.size == 0 or (raw_m.ndim == 2 and raw_m.shape[1] == 0):
            M_arr = np.empty((n_rows, len(domain.metas)), dtype=object)
            M_arr[:, :] = ""
        else:
            M_arr = _as_2d_array(raw_m, n_rows=n_rows, dtype=object)

    table = Table.from_numpy(domain, X=X_arr, Y=Y_arr, metas=M_arr)
    table.name = name
    return table


def records_to_orange_table(
    records: Sequence[Mapping[str, Any]],
    *,
    class_column: str | None = None,
    meta_columns: Sequence[str] | None = None,
    attribute_columns: Sequence[str] | None = None,
    numeric_as_attributes: bool = True,
    name: str = "Data",
) -> Table | None:
    """Build an Orange Table from dictionaries using shared role inference.

    Parameters
    ----------
    records:
        Row dictionaries.
    class_column:
        Optional numeric target variable. If present, it is removed from attrs/metas
        and exported as a continuous Orange class variable.
    meta_columns:
        Columns forced to metas.
    attribute_columns:
        Columns forced to continuous attributes.
    numeric_as_attributes:
        If True, numeric non-meta columns become Orange attributes.
    name:
        Table name.
    """
    if not records:
        return None

    keys = list(dict.fromkeys(key for row in records for key in row.keys()))
    forced_metas = set(meta_columns or [])
    forced_attrs = set(attribute_columns or [])

    class_vars: list[ContinuousVariable] = []
    if class_column and class_column in keys:
        class_vars = [ContinuousVariable(class_column)]
        keys = [key for key in keys if key != class_column]

    value_map = {key: [row.get(key, "") for row in records] for key in keys}
    attr_keys: list[str] = []
    meta_keys: list[str] = []
    for key in keys:
        if key in forced_metas:
            meta_keys.append(key)
        elif key in forced_attrs:
            attr_keys.append(key)
        elif numeric_as_attributes and not looks_like_meta_key(key) and column_is_numeric(value_map[key]):
            attr_keys.append(key)
        else:
            meta_keys.append(key)

    existing_names: set[str] = set()
    attributes = [ContinuousVariable(_unique_variable_name(existing_names, key)) for key in attr_keys]
    class_vars_named = [ContinuousVariable(_unique_variable_name(existing_names, var.name)) for var in class_vars]
    metas = [StringVariable(_unique_variable_name(existing_names, key)) for key in meta_keys]

    domain = Domain(attributes, class_vars=class_vars_named, metas=metas)

    X = np.asarray(
        [[as_float_or_nan(row.get(key, "")) for key in attr_keys] for row in records],
        dtype=float,
    ) if attr_keys else np.empty((len(records), 0), dtype=float)

    if class_column and class_vars_named:
        Y = np.asarray([as_float_or_nan(row.get(class_column, "")) for row in records], dtype=float).reshape(-1, 1)
    else:
        Y = None

    M = np.asarray(
        [["" if is_missing(row.get(key, "")) else str(row.get(key, "")) for key in meta_keys] for row in records],
        dtype=object,
    ) if meta_keys else np.empty((len(records), 0), dtype=object)

    return safe_table_from_numpy(domain, X=X, Y=Y, metas=M, name=name)


def domain_role_summary(domain: Domain) -> dict[str, list[str]]:
    return {
        "attributes": [v.name for v in domain.attributes],
        "class_vars": [v.name for v in domain.class_vars],
        "metas": [v.name for v in domain.metas],
    }


def format_domain_role_summary(domain: Domain, *, max_items: int = 8) -> str:
    def fmt(items: Sequence[str]) -> str:
        if not items:
            return "none"
        shown = ", ".join(items[:max_items])
        if len(items) > max_items:
            shown += f", … (+{len(items) - max_items})"
        return shown

    summary = domain_role_summary(domain)
    return (
        "Output column roles:\n"
        f"Attributes: {fmt(summary['attributes'])}\n"
        f"Class variables: {fmt(summary['class_vars'])}\n"
        f"Metas: {fmt(summary['metas'])}"
    )


__all__ = [
    "as_float_or_nan",
    "column_is_numeric",
    "domain_role_summary",
    "format_domain_role_summary",
    "is_missing",
    "looks_like_meta_key",
    "records_to_orange_table",
    "safe_table_from_numpy",
]
