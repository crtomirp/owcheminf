from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from chem_inf_widgets.chemcore.descriptors.fingerprints import (
    compute_fingerprints_from_smiles,
)
from chem_inf_widgets.chemcore.services.mordred_descriptor_service import (
    MORDRED_AVAILABLE,
    MordredComputeConfig,
    MordredDescriptorService,
)
from chem_inf_widgets.chemcore.services.qsar_regression_service import (
    RDKit_DESCRIPTOR_NAMES,
    _rdkit_descriptor_row,
)
from chem_inf_widgets.chemcore.services.qsar_target_contract import (
    DEFAULT_QSAR_TARGET_COLUMN,
    infer_target_label_from_model,
    prediction_column_name_for_target,
)


_SMILES_COLUMN_CANDIDATES = (
    "canonical_smiles",
    "smiles",
    "smi",
    "canonicalsmiles",
    "input_smiles",
)
_FINGERPRINT_RE = re.compile(
    r"^(?:(morgan|rdkit|maccs|avalon|fp|bit|mfp|ecfp|fcfp)[_-]?)?0*(\d+)$",
    re.IGNORECASE,
)
_FINGERPRINT_TYPE_ALIASES = {
    None: "morgan",
    "fp": "morgan",
    "bit": "morgan",
    "mfp": "morgan",
    "ecfp": "morgan",
    "fcfp": "morgan",
}
_MORDRED_DESCRIPTOR_NAMES: Optional[set[str]] = None


@dataclass(frozen=True)
class QSARPredictionModelBundle:
    """Prediction-ready model wrapper with feature-engineering metadata."""

    model: Any
    feature_names: tuple[str, ...] = ()
    target_label: Optional[str] = None
    recipe_kind: Optional[str] = None
    fingerprint_type: Optional[str] = None
    fingerprint_radius: int = 2
    fingerprint_n_bits: Optional[int] = None
    model_name: Optional[str] = None
    source_widget: Optional[str] = None
    training_rows: Optional[int] = None
    selected_feature_names: tuple[str, ...] = ()
    training_summary: dict[str, Any] = field(default_factory=dict)
    bundle_version: str = "2026.05"

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)

    def __getattr__(self, name: str):
        model = self.__dict__.get("model", None)
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    def __getitem__(self, item):
        return self.model[item]

    @property
    def feature_names_in_(self):
        if self.feature_names:
            return np.asarray(list(self.feature_names), dtype=object)
        return getattr(self.model, "feature_names_in_", None)


def build_qsar_prediction_bundle(
    model: Any,
    *,
    feature_names: Optional[list[str]] = None,
    target_label: Optional[str] = None,
    recipe_kind: Optional[str] = None,
    fingerprint_type: Optional[str] = None,
    fingerprint_radius: int = 2,
    fingerprint_n_bits: Optional[int] = None,
    model_name: Optional[str] = None,
    source_widget: Optional[str] = None,
    training_rows: Optional[int] = None,
    selected_feature_names: Optional[list[str]] = None,
    training_summary: Optional[dict[str, Any]] = None,
) -> QSARPredictionModelBundle:
    base_model = _unwrap_prediction_model(model)
    features = tuple(str(x) for x in (feature_names or _expected_features_from_model(model) or []))
    inferred_recipe, inferred_fp_type, inferred_n_bits = _infer_prediction_recipe(features)
    if isinstance(model, QSARPredictionModelBundle):
        current_model_name = model.model_name
        current_source_widget = model.source_widget
        current_training_rows = model.training_rows
        current_selected = tuple(str(x) for x in model.selected_feature_names)
        current_training_summary = dict(model.training_summary or {})
        current_bundle_version = str(model.bundle_version or "2026.05")
    else:
        current_model_name = None
        current_source_widget = None
        current_training_rows = None
        current_selected = tuple()
        current_training_summary = {}
        current_bundle_version = "2026.05"
    return QSARPredictionModelBundle(
        model=base_model,
        feature_names=features,
        target_label=target_label,
        recipe_kind=recipe_kind or inferred_recipe,
        fingerprint_type=fingerprint_type or inferred_fp_type,
        fingerprint_radius=int(fingerprint_radius),
        fingerprint_n_bits=fingerprint_n_bits if fingerprint_n_bits is not None else inferred_n_bits,
        model_name=str(model_name or current_model_name or type(base_model).__name__),
        source_widget=str(source_widget or current_source_widget or ""),
        training_rows=int(training_rows if training_rows is not None else (current_training_rows or 0)) or None,
        selected_feature_names=tuple(
            str(x)
            for x in (
                selected_feature_names
                or current_selected
                or feature_names
                or _expected_features_from_model(model)
                or []
            )
        ),
        training_summary=_json_safe(training_summary if training_summary is not None else current_training_summary),
        bundle_version=current_bundle_version,
    )


@dataclass(frozen=True)
class QSARPredictionPackagerConfig:
    id_column: str = "compound_id"
    target_label: str = DEFAULT_QSAR_TARGET_COLUMN
    prediction_column: str = prediction_column_name_for_target(DEFAULT_QSAR_TARGET_COLUMN)
    drop_non_numeric_features: bool = True
    include_input_columns: bool = True
    source_mode: Optional[str] = None


@dataclass(frozen=True)
class QSARPredictionPackageResult:
    predictions: pd.DataFrame
    feature_report: pd.DataFrame
    package_manifest: dict[str, Any]
    failed_records: pd.DataFrame


def _unwrap_prediction_model(model: Any) -> Any:
    return model.model if isinstance(model, QSARPredictionModelBundle) else model


def _select_numeric_features(df: pd.DataFrame, id_column: str) -> list[str]:
    blocked = {id_column}
    cols: list[str] = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(str(col))
    return cols


def _expected_features_from_model(model: Any) -> Optional[list[str]]:
    if isinstance(model, QSARPredictionModelBundle) and model.feature_names:
        return [str(x) for x in model.feature_names]
    if hasattr(model, "x_names_after_preprocess"):
        return [str(x) for x in list(getattr(model, "x_names_after_preprocess"))]
    if hasattr(model, "feature_names_in_"):
        values = getattr(model, "feature_names_in_", None)
        if values is not None:
            return [str(x) for x in list(values)]
    try:
        final = model[-1]
        if hasattr(final, "feature_names_in_"):
            values = getattr(final, "feature_names_in_", None)
            if values is not None:
                return [str(x) for x in list(values)]
    except Exception:
        pass
    return None


def _prediction_target_label(model: Any, config: QSARPredictionPackagerConfig) -> str:
    return infer_target_label_from_model(
        model,
        configured=config.target_label,
        fallback=DEFAULT_QSAR_TARGET_COLUMN,
    )


def model_bundle_metadata(model: Any) -> dict[str, Any]:
    bundle = model if isinstance(model, QSARPredictionModelBundle) else None
    base_model = _unwrap_prediction_model(model)
    feature_names = _expected_features_from_model(model) or []
    selected_feature_names = [
        str(x)
        for x in (
            list(bundle.selected_feature_names)
            if bundle is not None and bundle.selected_feature_names
            else list(getattr(base_model, "selected_names", []) or [])
        )
    ]
    return {
        "bundle_version": str(getattr(bundle, "bundle_version", "legacy")),
        "bundle_model_name": str(
            getattr(bundle, "model_name", "")
            or getattr(base_model, "model_name", "")
            or type(base_model).__name__
        ),
        "bundle_source_widget": str(
            getattr(bundle, "source_widget", "")
            or type(base_model).__name__
        ),
        "bundle_training_rows": int(getattr(bundle, "training_rows", 0) or 0),
        "bundle_feature_count": int(len(feature_names)),
        "bundle_selected_feature_count": int(len(selected_feature_names)),
        "bundle_selected_feature_names": selected_feature_names,
        "bundle_target_label": str(getattr(bundle, "target_label", "") or ""),
        "bundle_recipe_kind": str(getattr(bundle, "recipe_kind", "") or ""),
        "bundle_fingerprint_type": str(getattr(bundle, "fingerprint_type", "") or ""),
        "bundle_fingerprint_radius": int(getattr(bundle, "fingerprint_radius", 0) or 0),
        "bundle_fingerprint_n_bits": int(getattr(bundle, "fingerprint_n_bits", 0) or 0),
        "bundle_training_summary_keys": sorted(list((getattr(bundle, "training_summary", {}) or {}).keys())),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def selected_feature_names_from_model(
    model: Any,
    *,
    fallback_features: Optional[list[str]] = None,
) -> list[str]:
    expected = [str(x) for x in (fallback_features or _expected_features_from_model(model) or [])]
    if isinstance(model, QSARPredictionModelBundle) and model.selected_feature_names:
        return [str(x) for x in model.selected_feature_names]
    direct = getattr(model, "selected_names", None)
    if direct:
        return [str(x) for x in list(direct)]

    if hasattr(model, "pipelines"):
        union_names: list[str] = []
        seen: set[str] = set()
        for pipeline in list(getattr(model, "pipelines", []) or []):
            for name in selected_feature_names_from_model(pipeline, fallback_features=expected):
                if name not in seen:
                    seen.add(name)
                    union_names.append(name)
        return union_names or expected

    named_steps = getattr(model, "named_steps", None)
    if named_steps and expected:
        selector = named_steps.get("selector") or named_steps.get("feature_selection")
        if selector is not None and hasattr(selector, "get_support"):
            try:
                mask = np.asarray(selector.get_support(), dtype=bool).ravel()
            except Exception:
                mask = np.array([], dtype=bool)
            if mask.size == len(expected):
                return [name for name, keep in zip(expected, mask) if bool(keep)]
    return expected


def _fingerprint_spec_from_name(name: str) -> Optional[tuple[str, int]]:
    text = str(name).strip()
    if not text or text in RDKit_DESCRIPTOR_NAMES:
        return None
    match = _FINGERPRINT_RE.match(text)
    if match is None:
        return None
    raw_type, raw_index = match.groups()
    base_type = raw_type.lower() if raw_type else None
    fp_type = _FINGERPRINT_TYPE_ALIASES.get(base_type, base_type or "morgan")
    return fp_type, int(raw_index)


def _strip_legacy_source_prefix(name: str) -> str:
    text = str(name)
    return text[7:] if text.startswith("source_") else text


def _infer_prediction_recipe(
    feature_names: tuple[str, ...] | list[str],
) -> tuple[str, Optional[str], Optional[int]]:
    names = [_strip_legacy_source_prefix(str(x)) for x in feature_names]
    if not names:
        return "precomputed_table", None, None
    rdkit_names = [name for name in names if name in RDKit_DESCRIPTOR_NAMES]
    fp_specs = {name: _fingerprint_spec_from_name(name) for name in names}
    fp_names = [name for name, spec in fp_specs.items() if spec is not None]
    mordred_names = [name for name in names if name in _available_mordred_descriptor_names()]
    custom_names = [
        name
        for name in names
        if name not in rdkit_names and name not in fp_names and name not in mordred_names
    ]
    if custom_names:
        return "precomputed_table", None, None
    if not rdkit_names and not fp_names:
        if mordred_names and len(mordred_names) == len(names):
            return "mordred_selected", None, None
        return "precomputed_table", None, None
    fp_type = None
    fp_n_bits = None
    if fp_names:
        fp_types = {fp_specs[name][0] for name in fp_names if fp_specs[name] is not None}
        if len(fp_types) != 1:
            return "precomputed_table", None, None
        fp_type = next(iter(fp_types))
        fp_n_bits = (
            max(fp_specs[name][1] for name in fp_names if fp_specs[name] is not None) + 1
        )
    if rdkit_names and fp_names:
        return "rdkit_compact_plus_fingerprint", fp_type, fp_n_bits
    if rdkit_names:
        return "rdkit_compact", None, None
    if mordred_names and len(mordred_names) == len(names):
        return "mordred_selected", None, None
    return "fingerprint_only", fp_type, fp_n_bits


def _available_mordred_descriptor_names() -> set[str]:
    global _MORDRED_DESCRIPTOR_NAMES
    if _MORDRED_DESCRIPTOR_NAMES is not None:
        return _MORDRED_DESCRIPTOR_NAMES
    if not MORDRED_AVAILABLE:
        _MORDRED_DESCRIPTOR_NAMES = set()
        return _MORDRED_DESCRIPTOR_NAMES
    try:
        service = MordredDescriptorService(MordredComputeConfig(ignore_3d=True, nproc=1))
        _MORDRED_DESCRIPTOR_NAMES = {info.name for info in service.list_descriptors()}
    except Exception:
        _MORDRED_DESCRIPTOR_NAMES = set()
    return _MORDRED_DESCRIPTOR_NAMES


def _recipe_from_model(
    model: Any,
    expected_features: list[str],
) -> tuple[str, Optional[str], Optional[int], int]:
    if isinstance(model, QSARPredictionModelBundle):
        recipe_kind = str(model.recipe_kind or "").strip() or "precomputed_table"
        fp_type = str(model.fingerprint_type or "").strip() or None
        fp_n_bits = int(model.fingerprint_n_bits) if model.fingerprint_n_bits else None
        radius = int(model.fingerprint_radius or 2)
        return recipe_kind, fp_type, fp_n_bits, radius
    inferred_recipe, inferred_fp_type, inferred_n_bits = _infer_prediction_recipe(
        tuple(expected_features)
    )
    return inferred_recipe, inferred_fp_type, inferred_n_bits, 2


def _materialize_legacy_source_aliases(
    data: pd.DataFrame,
    expected_features: list[str],
) -> pd.DataFrame:
    for feature in expected_features:
        if feature in data.columns:
            continue
        alias = _strip_legacy_source_prefix(feature)
        if alias != feature and alias in data.columns:
            data[feature] = data[alias]
    return data


def _find_smiles_column(df: pd.DataFrame) -> Optional[str]:
    lowered = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in _SMILES_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _build_rdkit_feature_frame(
    smiles_list: list[str],
    expected_features: list[str],
    index,
) -> pd.DataFrame:
    rows = []
    for smiles in smiles_list:
        try:
            values = _rdkit_descriptor_row(smiles)
        except Exception:
            values = [np.nan] * len(RDKit_DESCRIPTOR_NAMES)
        rows.append(dict(zip(RDKit_DESCRIPTOR_NAMES, values)))
    desc_df = pd.DataFrame(rows, index=index)
    return desc_df.reindex(columns=expected_features)


def _build_fingerprint_feature_frame(
    smiles_list: list[str],
    expected_features: list[str],
    *,
    fp_type: str,
    n_bits: int,
    radius: int,
    index,
) -> tuple[pd.DataFrame, list[int]]:
    res = compute_fingerprints_from_smiles(
        smiles_list,
        fp_type=fp_type,
        bit_size=int(max(1, n_bits)),
        radius=int(radius),
        sanitize=True,
        remove_hs=True,
        remove_low_variance=False,
    )
    full = np.full((len(smiles_list), int(max(1, n_bits))), np.nan, dtype=float)
    if res.X.size and res.valid_indices:
        for row_i, src_i in enumerate(res.valid_indices):
            full[int(src_i), : res.X.shape[1]] = np.asarray(res.X[row_i], dtype=float)
    fp_df = pd.DataFrame(index=index)
    for name in expected_features:
        spec = _fingerprint_spec_from_name(name)
        if spec is None:
            fp_df[str(name)] = np.nan
            continue
        _, bit_index = spec
        if bit_index < full.shape[1]:
            fp_df[str(name)] = full[:, bit_index]
        else:
            fp_df[str(name)] = np.nan
    return fp_df, list(res.failed_indices)


def _engineer_missing_features(
    model: Any,
    data: pd.DataFrame,
    expected_features: list[str],
) -> tuple[pd.DataFrame, str, Optional[str], list[int]]:
    recipe_kind, fp_type, fp_n_bits, radius = _recipe_from_model(model, expected_features)
    if recipe_kind == "precomputed_table":
        return data, recipe_kind, None, []

    smiles_col = _find_smiles_column(data)
    if smiles_col is None:
        return data, recipe_kind, None, []

    smiles_values = [
        str(x).strip() if x is not None else ""
        for x in data[smiles_col].fillna("")
    ]
    feature_df = pd.DataFrame(index=data.index)
    failed_indices: list[int] = []
    stripped_expected = [_strip_legacy_source_prefix(name) for name in expected_features]

    if recipe_kind in {"rdkit_compact", "rdkit_compact_plus_fingerprint"}:
        rdkit_features = [name for name in stripped_expected if name in RDKit_DESCRIPTOR_NAMES]
        if rdkit_features:
            desc_df = _build_rdkit_feature_frame(smiles_values, rdkit_features, data.index)
            for col in desc_df.columns:
                feature_df[col] = desc_df[col]

    if recipe_kind in {"fingerprint_only", "rdkit_compact_plus_fingerprint"} and fp_type:
        fp_features = [
            name
            for name in stripped_expected
            if _fingerprint_spec_from_name(name) is not None
        ]
        if fp_features:
            max_required_bits = max(
                [
                    _fingerprint_spec_from_name(name)[1]
                    for name in fp_features
                    if _fingerprint_spec_from_name(name) is not None
                ]
                + [0]
            ) + 1
            fp_df, failed_indices = _build_fingerprint_feature_frame(
                smiles_values,
                fp_features,
                fp_type=fp_type,
                n_bits=int(max(fp_n_bits or 0, max_required_bits)),
                radius=radius,
                index=data.index,
            )
            for col in fp_df.columns:
                feature_df[col] = fp_df[col]

    if recipe_kind == "mordred_selected":
        if not MORDRED_AVAILABLE:
            return data, recipe_kind, "Mordred descriptors requested but mordred is not installed", []
        service = MordredDescriptorService(MordredComputeConfig(ignore_3d=True, nproc=1))
        mols_maybe, valid_idx = service.smiles_to_mols(smiles_values)
        valid_mols = [mols_maybe[i] for i in valid_idx if mols_maybe[i] is not None]
        mordred_valid = service.compute(valid_mols, stripped_expected)
        mordred_full = service.df_to_full_length(mordred_valid, valid_idx, len(smiles_values))
        for col in mordred_full.columns:
            feature_df[col] = mordred_full[col]
        failed_indices = [i for i, mol in enumerate(mols_maybe) if mol is None]

    if not feature_df.empty:
        for col in feature_df.columns:
            data[col] = feature_df[col]

    recipe_desc = {
        "rdkit_compact": "RDKit compact descriptor panel",
        "fingerprint_only": f"{str(fp_type).upper()} fingerprint bits",
        "rdkit_compact_plus_fingerprint": (
            f"RDKit compact descriptors + {str(fp_type).upper()} fingerprint bits"
        ),
        "mordred_selected": "Selected Mordred descriptors",
    }.get(recipe_kind)
    return data, recipe_kind, recipe_desc, failed_indices


def _predict_rows_with_fallback(
    model: Any,
    X_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, str]]:
    base_model = _unwrap_prediction_model(model)
    try:
        pred = np.asarray(base_model.predict(X_frame), dtype=float).ravel()
        return pred, {}
    except Exception as bulk_exc:
        pred = np.full(X_frame.shape[0], np.nan, dtype=float)
        failed: dict[int, str] = {}
        for row_index in range(X_frame.shape[0]):
            try:
                row_pred = np.asarray(
                    base_model.predict(X_frame.iloc[row_index : row_index + 1]),
                    dtype=float,
                ).ravel()
                pred[row_index] = float(row_pred[0])
            except Exception as row_exc:
                failed[row_index] = str(row_exc or bulk_exc)
        return pred, failed


def predict_with_qsar_model(
    model: Any,
    df: pd.DataFrame,
    config: QSARPredictionPackagerConfig | None = None,
) -> QSARPredictionPackageResult:
    config = config or QSARPredictionPackagerConfig()
    if df is None or df.empty:
        raise ValueError("Input data table is empty.")

    data = df.copy()
    if config.id_column not in data.columns:
        data[config.id_column] = [f"compound_{i + 1:04d}" for i in range(len(data))]

    candidate_features = _select_numeric_features(data, config.id_column)
    expected = _expected_features_from_model(model)
    missing: list[str] = []
    extra: list[str] = []
    recipe_kind = "precomputed_table"
    recipe_description = None

    if expected:
        data, recipe_kind, recipe_description, _ = _engineer_missing_features(
            model,
            data,
            expected,
        )
        data = _materialize_legacy_source_aliases(data, expected)
        candidate_features = _select_numeric_features(data, config.id_column)
        missing = [f for f in expected if f not in data.columns]
        extra = [f for f in candidate_features if f not in expected]
        feature_cols = [f for f in expected if f in data.columns]
    else:
        feature_cols = candidate_features

    if missing:
        raise ValueError(
            "The query table is missing model feature columns: "
            + ", ".join(missing[:20])
        )
    if not feature_cols:
        raise ValueError("No numeric feature columns are available for prediction.")

    X_frame = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    pred, row_failures = _predict_rows_with_fallback(model, X_frame)
    prediction_target_label = _prediction_target_label(model, config)

    if config.include_input_columns:
        out = data.copy()
    else:
        out = pd.DataFrame(index=data.index)
        if config.id_column in data.columns:
            out[config.id_column] = data[config.id_column].astype(str)
    out[config.prediction_column] = pred
    out["prediction_target_label"] = prediction_target_label
    out["prediction_status"] = [
        "failed" if i in row_failures else "predicted"
        for i in range(len(out))
    ]

    feature_report = pd.DataFrame(
        [
            {
                "feature": f,
                "status": "used",
                "missing_fraction": float(
                    pd.to_numeric(data[f], errors="coerce").isna().mean()
                ),
            }
            for f in feature_cols
        ]
        + [
            {
                "feature": f,
                "status": "extra_not_used",
                "missing_fraction": np.nan,
            }
            for f in extra
        ]
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_label": prediction_target_label,
        "prediction_column": config.prediction_column,
        "rows_input": int(len(data)),
        "rows_predicted": int(np.count_nonzero(np.isfinite(pred))),
        "rows_failed": int(len(row_failures)),
        "features_used": int(len(feature_cols)),
        "features_extra_not_used": int(len(extra)),
        "model_class": type(_unwrap_prediction_model(model)).__name__,
        "model_type": type(_unwrap_prediction_model(model)).__name__,
        "strict_feature_matching": bool(expected),
        "feature_names": list(feature_cols),
        "recipe_kind": recipe_kind,
        "recipe_description": recipe_description
        or "Precomputed numeric descriptor table",
        "auto_feature_engineering_used": bool(
            expected and recipe_kind != "precomputed_table"
        ),
        "source_mode": str(
            config.source_mode
            or ("smiles_table" if _find_smiles_column(data) else "numeric_table")
        ),
    }
    manifest.update(model_bundle_metadata(model))

    failed_rows = []
    smiles_col = _find_smiles_column(data)
    for row_index, reason in row_failures.items():
        failed_row = {
            config.id_column: str(
                data.iloc[row_index].get(config.id_column, f"compound_{row_index + 1:04d}")
            ),
            "reason": str(reason),
        }
        if smiles_col:
            failed_row["smiles"] = str(data.iloc[row_index].get(smiles_col, ""))
        failed_rows.append(failed_row)
    failed_columns = [config.id_column, "reason"] + (["smiles"] if smiles_col else [])
    failed = pd.DataFrame(failed_rows, columns=failed_columns)
    return QSARPredictionPackageResult(
        predictions=out,
        feature_report=feature_report,
        package_manifest=manifest,
        failed_records=failed,
    )


def save_model_pickle(model: Any, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f)
    return str(p)


def load_model_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def write_prediction_package(
    result: QSARPredictionPackageResult,
    out_prefix: str | Path,
    *,
    model: Any | None = None,
) -> dict[str, str]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": str(prefix.with_suffix(".predictions.csv")),
        "feature_report": str(prefix.with_suffix(".feature_report.csv")),
        "manifest": str(prefix.with_suffix(".manifest.json")),
        "failed": str(prefix.with_suffix(".failed.csv")),
    }
    result.predictions.to_csv(paths["predictions"], index=False)
    result.feature_report.to_csv(paths["feature_report"], index=False)
    result.failed_records.to_csv(paths["failed"], index=False)
    Path(paths["manifest"]).write_text(
        json.dumps(result.package_manifest, indent=2),
        encoding="utf-8",
    )
    if model is not None:
        paths["model_pickle"] = save_model_pickle(model, prefix.with_suffix(".model.pkl"))
    return paths


def write_model_bundle_package(
    model: Any,
    out_prefix: str | Path,
) -> dict[str, str]:
    prefix = Path(out_prefix)
    if prefix.suffix.lower() == ".pkl":
        prefix = prefix.with_suffix("")
    if prefix.suffix.lower() == ".model":
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bundle = model if isinstance(model, QSARPredictionModelBundle) else build_qsar_prediction_bundle(model)
    feature_names = [str(x) for x in bundle.feature_names]
    selected_features = selected_feature_names_from_model(bundle, fallback_features=feature_names)
    manifest = {
        "artifact_kind": "qsar_prediction_model_bundle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle_version": str(bundle.bundle_version or "2026.05"),
        "model_name": str(bundle.model_name or type(bundle.model).__name__),
        "source_widget": str(bundle.source_widget or ""),
        "target_label": str(bundle.target_label or ""),
        "recipe_kind": str(bundle.recipe_kind or ""),
        "fingerprint_type": str(bundle.fingerprint_type or ""),
        "fingerprint_radius": int(bundle.fingerprint_radius or 2),
        "fingerprint_n_bits": int(bundle.fingerprint_n_bits or 0),
        "training_rows": int(bundle.training_rows or 0),
        "feature_count": int(len(feature_names)),
        "selected_feature_count": int(len(selected_features)),
        "feature_names": feature_names,
        "selected_feature_names": selected_features,
        "training_summary": _json_safe(bundle.training_summary),
        "prediction_packager_metadata": _json_safe(model_bundle_metadata(bundle)),
    }
    paths = {
        "model_pickle": save_model_pickle(bundle, prefix.with_suffix(".model.pkl")),
        "manifest_json": str(prefix.with_suffix(".manifest.json")),
        "feature_names_txt": str(prefix.with_suffix(".features.txt")),
        "selected_features_txt": str(prefix.with_suffix(".selected_features.txt")),
    }
    Path(paths["manifest_json"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    Path(paths["feature_names_txt"]).write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    Path(paths["selected_features_txt"]).write_text("\n".join(selected_features) + "\n", encoding="utf-8")
    return paths
