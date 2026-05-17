"""QSAR dataset builder service.

This module prepares QSAR-ready activity tables from raw or semi-curated
bioactivity records.  It is intentionally independent of Orange widgets so it
can be used from the GUI, CLI, tests, and notebooks.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # RDKit is expected in the add-on, but keep import failure explicit.
    from rdkit import Chem
    from rdkit.Chem import inchi
except Exception:  # pragma: no cover - environment dependent
    Chem = None
    inchi = None


DEFAULT_SMILES_COLUMNS = (
    "smiles", "canonical_smiles", "standardized_smiles", "canonical_smiles_std",
    "mol_smiles", "structure", "smi", "smiles_string", "rdkit_smiles",
    "isomeric_smiles", "iso_smiles", "parent_smiles", "inchi_smiles",
    "molecule_smiles", "compound_smiles", "ligand_smiles",
    # PubChem
    "isomericsmiles", "canonicalsmiles",
    # BindingDB
    "ligand_smiles", "smiles (rdkit notation)",
    # ExCAPE-DB
    "smiles_standardized",
    # ZINC / Enamine
    "zinc_smiles", "smiles_zinc",
)
DEFAULT_NAME_COLUMNS = (
    "name", "compound_name", "molecule_name", "molecule_id",
    "chembl_id", "compound_id", "cmpd_id", "cpd_id", "id",
    "mol_id", "molregno", "pubchem_cid", "cid",
    "chembl_molecule_chembl_id", "parent_molecule_chembl_id",
    # BindingDB
    "bindingdb_monomer_id", "ligand_name",
    # ExCAPE-DB
    "original_entry_id", "inchi_key_std",
    # PubChem
    "cid", "sid", "iupac_name",
    # ZINC
    "zinc_id", "zinc15_id",
)
DEFAULT_ACTIVITY_COLUMNS = (
    # pre-converted pActivity
    "pchembl_value", "pactivity", "p_activity",
    "pic50", "pic_50", "p_ic50",
    "pki", "p_ki", "pec50", "p_ec50",
    "pkd", "p_kd", "pkb", "p_kb",
    "pcc50", "p_cc50",
    # raw values
    "standard_value", "activity_value", "value",
    "ic50", "ic_50", "ki", "kd", "ec50", "ec_50",
    "cc50", "lc50", "inhibition", "percent_inhibition",
    # BindingDB
    "ki (nm)", "ic50 (nm)", "kd (nm)", "ec50 (nm)",
    "ki_nm", "ic50_nm",
    # ExCAPE-DB
    "activity",
)
DEFAULT_UNIT_COLUMNS = (
    "standard_units", "units", "unit", "activity_units",
    "assay_units", "concentration_units",
    # ChEMBL
    "standard_units",
    # BindingDB
    "affinity_units",
)
DEFAULT_RELATION_COLUMNS = (
    "standard_relation", "relation", "activity_relation",
    "operator", "qualifier", "comparison",
    # ChEMBL
    "standard_relation",
    # PubChem
    "activity_outcome",
)
DEFAULT_TYPE_COLUMNS = (
    "standard_type", "endpoint", "activity_type", "type",
    "assay_type", "measurement_type", "assay_endpoint",
    # ChEMBL
    "standard_type",
    # BindingDB
    "affinity_type",
    # ExCAPE-DB
    "activity_id",
)

# Known SMILES fragment patterns for content-based detection
_SMILES_CHARS = set("CNOSPFClBrInH@+\\/#=[]()0123456789.")

# Known unit strings (lowercase)
_KNOWN_UNITS = frozenset({
    "nm", "um", "µm", "mm", "m", "pm",
    "nM", "uM", "µM", "mM", "M", "pM",
    "mg/ml", "mg/l", "g/l", "ug/ml", "µg/ml",
    "% inhibition", "%", "log(m)",
})

# Known relation strings
_KNOWN_RELATIONS = frozenset({"=", ">", "<", ">=", "<=", "~", ">>", "<<"})


@dataclass(frozen=True)
class QSARDatasetBuilderConfig:
    smiles_column: str | None = None
    name_column: str | None = None
    activity_column: str | None = None
    unit_column: str | None = None
    relation_column: str | None = None
    endpoint_column: str | None = None
    target_endpoint: str = ""  # e.g. IC50; empty accepts all endpoints
    target_unit: str = "nM"
    relation_policy: str = "exact_only"  # exact_only, allow_inequalities
    aggregation: str = "median"  # median, mean, min, max, first
    duplicate_key: str = "standard_inchikey"  # standard_inchikey, canonical_smiles, raw_smiles
    min_pactivity: float | None = None
    max_pactivity: float | None = None
    keep_rejected: bool = True


@dataclass(frozen=True)
class QSARDatasetBuilderResult:
    prepared_records: list[dict[str, Any]]
    rejected_records: list[dict[str, Any]]
    curation_report: list[dict[str, Any]]
    summary: dict[str, Any]
    detected_columns: dict[str, str | None]


def _norm_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


# ── Name-based detection ───────────────────────────────────────────────────

def detect_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    """Exact or substring name match. Returns first hit or None."""
    if not columns:
        return None
    normalized = {_norm_name(c): c for c in columns}
    for cand in candidates:
        key = _norm_name(cand)
        if key in normalized:
            return normalized[key]
    for col in columns:
        low = _norm_name(col)
        if any(_norm_name(c) in low for c in candidates):
            return col
    return None


def _name_score(col: str, candidates: Sequence[str]) -> float:
    """0.0–1.0 name-match confidence: exact=1.0, substring=0.6, none=0.0."""
    low = _norm_name(col)
    for cand in candidates:
        if _norm_name(cand) == low:
            return 1.0
    for cand in candidates:
        if _norm_name(cand) in low or low in _norm_name(cand):
            return 0.6
    return 0.0


# ── Content-based scoring ─────────────────────────────────────────────────

def _sample_values(records: Sequence[Mapping[str, Any]], col: str, n: int = 40) -> list:
    out = []
    for r in records:
        v = r.get(col)
        if v is not None and v != "":
            out.append(v)
        if len(out) >= n:
            break
    return out


def _content_smiles_score(values: list) -> float:
    """Fraction of non-empty values that look like SMILES."""
    if not values:
        return 0.0
    hits = 0
    for v in values:
        s = str(v).strip()
        if len(s) < 4:
            continue
        # SMILES must contain at least one letter and one of: ()=#
        has_letter = any(c.isalpha() for c in s)
        has_bond = any(c in "()=#" for c in s)
        char_frac = sum(1 for c in s if c in _SMILES_CHARS) / len(s)
        if has_letter and has_bond and char_frac > 0.7:
            hits += 1
    return hits / len(values)


def _content_numeric_score(values: list) -> float:
    """Fraction of values that are finite numbers."""
    if not values:
        return 0.0
    hits = sum(1 for v in values if _as_float_safe(v) is not None)
    return hits / len(values)


def _content_pactivity_score(values: list) -> float:
    """High score if numeric values fall in [0, 15] (pActivity range)."""
    nums = [_as_float_safe(v) for v in values if _as_float_safe(v) is not None]
    if not nums:
        return 0.0
    in_range = sum(1 for x in nums if 0.0 <= x <= 15.0)
    return (in_range / len(nums)) * _content_numeric_score(values)


def _content_raw_activity_score(values: list) -> float:
    """High if numeric values span typical nM/µM bioassay range (1e-3 – 1e8)."""
    nums = [_as_float_safe(v) for v in values if _as_float_safe(v) is not None]
    if not nums:
        return 0.0
    in_range = sum(1 for x in nums if 1e-4 <= x <= 1e8)
    return (in_range / len(nums)) * _content_numeric_score(values)


def _content_unit_score(values: list) -> float:
    """Fraction of string values that match known unit tokens."""
    if not values:
        return 0.0
    hits = sum(
        1 for v in values
        if str(v).strip().lower() in {u.lower() for u in _KNOWN_UNITS}
    )
    return hits / len(values)


def _content_relation_score(values: list) -> float:
    """Fraction of values that match known relation operators."""
    if not values:
        return 0.0
    hits = sum(1 for v in values if str(v).strip() in _KNOWN_RELATIONS)
    return hits / len(values)


def _as_float_safe(v: Any) -> float | None:
    try:
        f = float(v)
        return f if not math.isnan(f) else None
    except Exception:
        return None


# ── Smart ranked detection ─────────────────────────────────────────────────

def _detect_ranked(
    columns: Sequence[str],
    name_candidates: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    content_scorer,           # callable(values) → 0..1
    content_weight: float = 0.5,
    excluded: set[str] | None = None,
) -> tuple[str | None, float]:
    """
    Returns (best_column, confidence) picking the column that maximises
    name_score * (1 - content_weight) + content_score * content_weight.
    """
    if not columns:
        return None, 0.0
    best_col, best_score = None, 0.0
    for col in columns:
        if excluded and col in excluded:
            continue
        ns = _name_score(col, name_candidates)
        if records:
            vals = _sample_values(records, col)
            cs = content_scorer(vals)
        else:
            cs = 0.0
        score = ns * (1 - content_weight) + cs * content_weight
        if score > best_score:
            best_score, best_col = score, col
    return best_col, best_score


# ── Full auto-detection ────────────────────────────────────────────────────

def smart_detect_columns(
    columns: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Returns dict with keys matching QSARDatasetBuilderConfig field names plus
    extra diagnostic info:
      column_name → detected column (str | None)
      column_name + '_confidence' → 0..1 float
      column_name + '_method' → 'name' | 'content' | 'none'
      'is_pactivity' → True if activity column already contains pActivity values
      'suggested_endpoint' → most common value in endpoint column (str | None)
      'suggested_unit' → most common value in unit column (str | None)
    """
    used: set[str] = set()
    result: dict[str, Any] = {}

    def _pick(field: str, name_cands, scorer, cw=0.5):
        col, conf = _detect_ranked(columns, name_cands, records, scorer,
                                   content_weight=cw, excluded=used)
        if col and conf > 0.05:
            used.add(col)
            result[field] = col
            result[f"{field}_confidence"] = round(conf, 3)
            result[f"{field}_method"] = "name" if _name_score(col, name_cands) >= 0.6 else "content"
        else:
            result[field] = None
            result[f"{field}_confidence"] = 0.0
            result[f"{field}_method"] = "none"

    # SMILES — content heavily weighted (SMILES strings are distinctive)
    _pick("smiles_column", DEFAULT_SMILES_COLUMNS, _content_smiles_score, cw=0.7)

    # Name/ID — no reliable content signal, name-only
    _pick("name_column", DEFAULT_NAME_COLUMNS, lambda v: 0.0, cw=0.0)

    # Activity — check if values are pActivity range OR raw assay range
    # Try pActivity columns first (pchembl_value, pactivity, etc.)
    pact_col, pact_conf = _detect_ranked(
        columns, DEFAULT_ACTIVITY_COLUMNS[:6], records, _content_pactivity_score, 0.5, used)
    raw_col, raw_conf = _detect_ranked(
        columns, DEFAULT_ACTIVITY_COLUMNS, records, _content_raw_activity_score, 0.5, used)

    if pact_col and pact_conf >= raw_conf:
        used.add(pact_col)
        result["activity_column"] = pact_col
        result["activity_column_confidence"] = round(pact_conf, 3)
        result["activity_column_method"] = "name" if _name_score(pact_col, DEFAULT_ACTIVITY_COLUMNS) >= 0.6 else "content"
        # Detect if already pActivity (no unit conversion needed)
        vals = _sample_values(records, pact_col)
        result["is_pactivity"] = _content_pactivity_score(vals) > 0.6
    elif raw_col:
        used.add(raw_col)
        result["activity_column"] = raw_col
        result["activity_column_confidence"] = round(raw_conf, 3)
        result["activity_column_method"] = "name" if _name_score(raw_col, DEFAULT_ACTIVITY_COLUMNS) >= 0.6 else "content"
        result["is_pactivity"] = False
    else:
        result["activity_column"] = None
        result["activity_column_confidence"] = 0.0
        result["activity_column_method"] = "none"
        result["is_pactivity"] = False

    _pick("unit_column",     DEFAULT_UNIT_COLUMNS,     _content_unit_score,     cw=0.6)
    _pick("relation_column", DEFAULT_RELATION_COLUMNS, _content_relation_score, cw=0.6)
    _pick("endpoint_column", DEFAULT_TYPE_COLUMNS,     lambda v: 0.0,           cw=0.0)

    # Suggest most-common endpoint and unit from actual data
    result["suggested_endpoint"] = _most_common_value(records, result.get("endpoint_column"))
    result["suggested_unit"]     = _most_common_value(records, result.get("unit_column"))

    return result


def _most_common_value(records: Sequence[Mapping[str, Any]], col: str | None) -> str | None:
    if not col or not records:
        return None
    from collections import Counter
    vals = [str(r.get(col, "")).strip() for r in records if str(r.get(col, "")).strip()]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def auto_detect_columns(records: Sequence[Mapping[str, Any]], config: QSARDatasetBuilderConfig) -> dict[str, str | None]:
    columns = list(records[0].keys()) if records else []
    det = smart_detect_columns(columns, records)
    return {
        "smiles_column":   config.smiles_column   or det.get("smiles_column"),
        "name_column":     config.name_column      or det.get("name_column"),
        "activity_column": config.activity_column  or det.get("activity_column"),
        "unit_column":     config.unit_column      or det.get("unit_column"),
        "relation_column": config.relation_column  or det.get("relation_column"),
        "endpoint_column": config.endpoint_column  or det.get("endpoint_column"),
    }


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        if math.isnan(float(value)):
            return None
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _unit_to_molar(value: float, unit: str) -> float | None:
    unit_norm = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    factors = {
        "m": 1.0,
        "mol/l": 1.0,
        "mol l-1": 1.0,
        "mm": 1e-3,
        "millimolar": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "μm": 1e-6,
        "micromolar": 1e-6,
        "nm": 1e-9,
        "nanomolar": 1e-9,
        "pm": 1e-12,
        "picomolar": 1e-12,
    }
    factor = factors.get(unit_norm)
    if factor is None:
        return None
    return value * factor


def _looks_like_pactivity(col_name: str) -> bool:
    low = _norm_name(col_name)
    return low in {"pchembl_value", "pactivity", "pic50", "pki", "pkd", "pec50"} or low.startswith("p_")


def compute_pactivity(value: Any, activity_col: str, unit: str | None, default_unit: str) -> tuple[float | None, str]:
    numeric = _as_float(value)
    if numeric is None:
        return None, "activity_not_numeric"
    if _looks_like_pactivity(activity_col):
        return numeric, "pactivity_direct"
    molar = _unit_to_molar(numeric, unit or default_unit)
    if molar is None:
        return None, "unsupported_or_missing_unit"
    if molar <= 0:
        return None, "activity_not_positive"
    return -math.log10(molar), "converted_from_concentration"


def _canonical_identity(smiles: str, mode: str) -> tuple[str, str, str]:
    """Return (key, canonical_smiles, identity_status)."""
    if mode == "raw_smiles" or Chem is None:
        return smiles, smiles, "raw_smiles"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, smiles, "invalid_for_identity"
    try:
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        can = smiles
    if mode == "canonical_smiles":
        return can, can, "canonical_smiles"
    if mode == "standard_inchikey" and inchi is not None:
        try:
            ik = inchi.MolToInchiKey(mol)
            if ik:
                return ik, can, "standard_inchikey"
        except Exception:
            pass
    return can, can, "canonical_smiles_fallback"


def _aggregate(values: Sequence[float], method: str) -> float:
    arr = np.asarray(values, dtype=float)
    if method == "mean":
        return float(np.nanmean(arr))
    if method == "min":
        return float(np.nanmin(arr))
    if method == "max":
        return float(np.nanmax(arr))
    if method == "first":
        return float(arr[0])
    return float(np.nanmedian(arr))


def _unique_group_texts(group: pd.DataFrame, column: str) -> list[str]:
    if column not in group.columns:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for value in group[column].values:
        text = _as_str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _merge_group_pipe_tokens(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    seen: set[str] = set()
    out: list[str] = []
    for value in group[column].values:
        for token in str(value or "").split(" | "):
            text = token.strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return " | ".join(out)


def build_qsar_dataset(
    records: Sequence[Mapping[str, Any]],
    config: QSARDatasetBuilderConfig | None = None,
) -> QSARDatasetBuilderResult:
    config = config or QSARDatasetBuilderConfig()
    rows = [dict(r) for r in records]
    detected = auto_detect_columns(rows, config)
    smiles_col = detected["smiles_column"]
    activity_col = detected["activity_column"]
    unit_col = detected["unit_column"]
    relation_col = detected["relation_column"]
    endpoint_col = detected["endpoint_column"]
    name_col = detected["name_column"]

    if not smiles_col:
        raise ValueError("No SMILES-like column was detected. Specify smiles_column.")
    if not activity_col:
        raise ValueError("No activity/pActivity column was detected. Specify activity_column.")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []

    target_endpoint = config.target_endpoint.strip().lower()

    for idx, row in enumerate(rows):
        original_id = _as_str(row.get(name_col)) if name_col else f"row_{idx+1}"
        reasons: list[str] = []
        smiles = _as_str(row.get(smiles_col))
        if not smiles:
            reasons.append("missing_smiles")

        endpoint = _as_str(row.get(endpoint_col)) if endpoint_col else ""
        if target_endpoint and endpoint.strip().lower() != target_endpoint:
            reasons.append("endpoint_not_selected")

        relation = _as_str(row.get(relation_col)) if relation_col else "="
        if config.relation_policy == "exact_only" and relation and relation not in {"=", "'='"}:
            reasons.append("non_exact_relation")

        unit = _as_str(row.get(unit_col)) if unit_col else config.target_unit
        pactivity, conversion_status = compute_pactivity(row.get(activity_col), activity_col, unit, config.target_unit)
        if pactivity is None:
            reasons.append(conversion_status)
        else:
            if config.min_pactivity is not None and pactivity < config.min_pactivity:
                reasons.append("below_min_pactivity")
            if config.max_pactivity is not None and pactivity > config.max_pactivity:
                reasons.append("above_max_pactivity")

        identity_key = ""
        canonical_smiles = smiles
        identity_status = "not_computed"
        if smiles:
            identity_key, canonical_smiles, identity_status = _canonical_identity(smiles, config.duplicate_key)
            if identity_status == "invalid_for_identity":
                reasons.append("invalid_smiles")

        base_record = {
            "row_index": idx,
            "compound_id": original_id,
            "smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "identity_key": identity_key,
            "identity_status": identity_status,
            "endpoint": endpoint,
            "relation": relation,
            "activity_value": row.get(activity_col),
            "activity_unit": unit,
            "pActivity_raw": pactivity if pactivity is not None else np.nan,
            "conversion_status": conversion_status,
            "reject_reasons": ";".join(reasons),
        }
        for key, value in row.items():
            if key not in base_record and key not in {smiles_col, activity_col}:
                base_record[f"source_{key}"] = value

        report.append({**base_record, "accepted_before_aggregation": not reasons})
        if reasons:
            rejected.append(base_record)
        else:
            accepted.append(base_record)

    prepared: list[dict[str, Any]] = []
    duplicate_groups = 0
    if accepted:
        df = pd.DataFrame(accepted)
        group_key = "identity_key"
        for identity_key, group in df.groupby(group_key, dropna=False, sort=False):
            vals = [float(v) for v in group["pActivity_raw"].values]
            p_activity = _aggregate(vals, config.aggregation)
            first = group.iloc[0].to_dict()
            n = len(group)
            if n > 1:
                duplicate_groups += 1
            source_row_ids = _unique_group_texts(group, "source_row_id")
            source_transform_logs = _unique_group_texts(group, "source_transform_log")
            source_qc_flags_all = _merge_group_pipe_tokens(group, "source_qc_flags")
            source_dropped_reasons = _unique_group_texts(group, "source_dropped_reason")
            prepared.append(
                {
                    **first,
                    "pActivity": p_activity,
                    "n_measurements": int(n),
                    "pActivity_min": float(np.nanmin(vals)),
                    "pActivity_max": float(np.nanmax(vals)),
                    "pActivity_std": float(np.nanstd(vals)) if n > 1 else 0.0,
                    "aggregation_method": config.aggregation,
                    "duplicate_group": bool(n > 1),
                    "source_row_ids": ";".join(source_row_ids),
                    "source_transform_logs": " || ".join(source_transform_logs),
                    "source_qc_flags_all": source_qc_flags_all,
                    "source_dropped_reasons": " | ".join(source_dropped_reasons),
                }
            )

    summary = {
        "input_records": len(rows),
        "accepted_measurements_before_aggregation": len(accepted),
        "rejected_records": len(rejected),
        "prepared_compounds": len(prepared),
        "duplicate_groups": duplicate_groups,
        "smiles_column": smiles_col,
        "activity_column": activity_col,
        "unit_column": unit_col,
        "relation_column": relation_col,
        "endpoint_column": endpoint_col,
        "name_column": name_col,
        "target_endpoint": config.target_endpoint,
        "target_unit": config.target_unit,
        "relation_policy": config.relation_policy,
        "aggregation": config.aggregation,
        "duplicate_key": config.duplicate_key,
    }
    return QSARDatasetBuilderResult(prepared, rejected, report, summary, detected)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t").to_dict(orient="records")
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported QSAR dataset input format: {suffix}")


def write_result_files(result: QSARDatasetBuilderResult, out_prefix: str | Path, write_json: bool = True) -> dict[str, str]:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    files = {
        "prepared": str(prefix.with_suffix(".qsar_ready.csv")),
        "rejected": str(prefix.with_suffix(".rejected.csv")),
        "report": str(prefix.with_suffix(".curation_report.csv")),
        "summary": str(prefix.with_suffix(".summary.csv")),
    }
    pd.DataFrame(result.prepared_records).to_csv(files["prepared"], index=False)
    pd.DataFrame(result.rejected_records).to_csv(files["rejected"], index=False)
    pd.DataFrame(result.curation_report).to_csv(files["report"], index=False)
    pd.DataFrame([result.summary]).to_csv(files["summary"], index=False)
    if write_json:
        json_path = str(prefix.with_suffix(".summary.json"))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"summary": result.summary, "detected_columns": result.detected_columns}, f, indent=2)
        files["summary_json"] = json_path
    return files
