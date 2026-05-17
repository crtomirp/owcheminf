from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable, Optional

try:  # pragma: no cover - depends on runtime environment
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.result import ServiceIssue
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles

# Canonical field names used across import, QC, standardization and QSAR widgets.
MOL_ID = "mol_id"
NAME = "Name"
SMILES = "SMILES"
INPUT_SMILES = "input_smiles"
CANONICAL_SMILES = "canonical_smiles"
STANDARDIZED_SMILES = "standardized_smiles"
INCHI = "inchi"
INCHIKEY = "inchikey"
QC_STATUS = "qc_status"
QC_SEVERITY = "qc_severity"
QC_ISSUE_CODES = "qc_issue_codes"
QC_ISSUES = "qc_issues"
QC_N_ISSUES = "qc_n_issues"
QC_DUPLICATE_KEY = "qc_duplicate_key"
QC_DUPLICATE_COUNT = "qc_duplicate_count"
QC_VERSION_FIELD = "qc_version"
STANDARDIZATION_STATUS = "standardization_status"
STANDARDIZATION_PROFILE = "standardization_profile"
STANDARDIZATION_CHANGED = "standardization_changed"
STANDARDIZATION_LOG = "standardization_log"
STANDARDIZATION_INPUT_SMILES = "standardization_input_smiles"
STANDARDIZATION_OUTPUT_SMILES = "standardization_output_smiles"
STANDARDIZATION_VERSION_FIELD = "standardization_version"
SOURCE_FORMAT = "source_format"
SOURCE_ROW_INDEX = "source_row_index"
ROW_ID = "row_id"
TRANSFORM_LOG = "transform_log"
DROPPED_REASON = "dropped_reason"
QC_FLAGS = "qc_flags"

CURATION_STAGE = "curation_stage"
CURATION_STATUS = "curation_status"
CURATION_READY_FOR_QSAR = "curation_ready_for_qsar"
CURATION_READY_FOR_DOCKING = "curation_ready_for_docking"
CURATION_BLOCKERS = "curation_blockers"
CURATION_WARNINGS = "curation_warnings"
CURATION_RECOMMENDED_NEXT_STEP = "curation_recommended_next_step"
CURATION_VERSION_FIELD = "curation_version"

STRUCTURE_FIELDS = {
    SMILES,
    INPUT_SMILES,
    CANONICAL_SMILES,
    STANDARDIZED_SMILES,
    "SMILES_STD",
    "SMILES_ORIG",
    "INPUT_SMILES",
}
IDENTIFIER_FIELDS = {MOL_ID, NAME, "id", "ID", "compound_id", "molecule_id", INCHIKEY, INCHI}
META_FIELDS = STRUCTURE_FIELDS | IDENTIFIER_FIELDS | {
    QC_STATUS,
    QC_SEVERITY,
    QC_ISSUE_CODES,
    QC_ISSUES,
    QC_N_ISSUES,
    QC_DUPLICATE_KEY,
    QC_DUPLICATE_COUNT,
    QC_VERSION_FIELD,
    STANDARDIZATION_STATUS,
    STANDARDIZATION_PROFILE,
    STANDARDIZATION_CHANGED,
    STANDARDIZATION_LOG,
    STANDARDIZATION_INPUT_SMILES,
    STANDARDIZATION_OUTPUT_SMILES,
    STANDARDIZATION_VERSION_FIELD,
    SOURCE_FORMAT,
    SOURCE_ROW_INDEX,
    ROW_ID,
    TRANSFORM_LOG,
    DROPPED_REASON,
    QC_FLAGS,
    CURATION_STAGE,
    CURATION_STATUS,
    CURATION_READY_FOR_QSAR,
    CURATION_READY_FOR_DOCKING,
    CURATION_BLOCKERS,
    CURATION_WARNINGS,
    CURATION_RECOMMENDED_NEXT_STEP,
    CURATION_VERSION_FIELD,
    "IMPORT_SOURCE_FORMAT",
    "IMPORT_ROW_INDEX",
    "IMPORT_WARNINGS",
    "STD_LOG",
    "STD_PROFILE",
    "STD_OK",
    "STD_INPUT_SMILES",
    "STD_OUTPUT_SMILES",
    "STD_CHANGED",
    "STD_STEPS",
}


@dataclass(frozen=True)
class MoleculeContractRecord:
    """Lightweight, auditable molecule row used by services and reports."""

    row_index: int
    mol_id: str
    name: str
    input_smiles: str
    canonical_smiles: str
    standardized_smiles: str = ""
    inchikey: str = ""
    source_format: str = ""
    props: dict[str, Any] = field(default_factory=dict)


def is_meta_field(name: str) -> bool:
    text = str(name or "").strip()
    lower = text.lower()
    return text in META_FIELDS or lower in {f.lower() for f in META_FIELDS} or lower.endswith("_smiles") or lower.endswith("_status")


def _inchi_key(mol: Any) -> str:
    if Chem is None or mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(Chem.Mol(mol)) or ""
    except Exception:
        return ""


def _stable_row_id(
    *,
    source_format: str,
    row_index: int,
    mol_id: str,
    input_smiles: str,
) -> str:
    payload = "|".join(
        [
            str(source_format or ""),
            str(int(row_index or 0)),
            str(mol_id or ""),
            str(input_smiles or ""),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _props_dict(target: ChemMol | dict[str, Any]) -> dict[str, Any]:
    if isinstance(target, ChemMol):
        props = target.props if isinstance(target.props, dict) else {}
        target.props = props
        return props
    return target


def append_transform_step(target: ChemMol | dict[str, Any], step: str) -> str:
    """Append a pipeline step to the shared transform log."""

    step_text = str(step or "").strip()
    if not step_text:
        return ""

    props = _props_dict(target)
    existing = str(props.get(TRANSFORM_LOG, "") or "").strip()
    parts = [part.strip() for part in existing.split(" | ") if part.strip()]
    if not parts or parts[-1] != step_text:
        parts.append(step_text)
    props[TRANSFORM_LOG] = " | ".join(parts)
    return props[TRANSFORM_LOG]


def _append_pipe_token(target: ChemMol | dict[str, Any], field: str, token: str) -> str:
    token_text = str(token or "").strip()
    if not token_text:
        return ""

    props = _props_dict(target)
    existing = str(props.get(field, "") or "").strip()
    parts = [part.strip() for part in existing.split(" | ") if part.strip()]
    if token_text not in parts:
        parts.append(token_text)
    props[field] = " | ".join(parts)
    return props[field]


def append_qc_flag(target: ChemMol | dict[str, Any], flag: str) -> str:
    """Append a QC flag to the shared contract field."""

    return _append_pipe_token(target, QC_FLAGS, flag)


def set_dropped_reason(target: ChemMol | dict[str, Any], reason: str) -> str:
    """Record why a row was rejected or dropped from downstream use."""

    reason_text = str(reason or "").strip()
    if not reason_text:
        return ""

    props = _props_dict(target)
    props[DROPPED_REASON] = reason_text
    return reason_text


def ensure_contract_props(
    cm: ChemMol,
    *,
    row_index: int = 0,
    source_format: str = "",
    input_smiles: str = "",
    mol_id: str = "",
) -> ChemMol:
    """Return ``cm`` with canonical contract properties filled when possible.

    The function mutates ``cm.props`` intentionally because ChemMol is the shared
    in-memory object used by Orange widget outputs. It does not change chemistry.
    """

    props = cm.props if isinstance(cm.props, dict) else {}
    cm.props = props
    canonical = props.get(CANONICAL_SMILES) or props.get(SMILES) or safe_canonical_smiles(cm.mol, remove_hs=True)
    in_smi = input_smiles or props.get(INPUT_SMILES) or props.get("INPUT_SMILES") or props.get(SMILES) or canonical
    key = props.get(INCHIKEY) or _inchi_key(cm.mol)
    record_id = mol_id or props.get(MOL_ID) or props.get("compound_id") or props.get("molecule_id") or cm.name or key or f"mol_{row_index}"
    final_source_format = source_format or str(props.get(SOURCE_FORMAT, props.get("IMPORT_SOURCE_FORMAT", "")) or "")
    final_row_index = int(props.get(SOURCE_ROW_INDEX) or row_index or 0)

    props.setdefault(MOL_ID, str(record_id))
    props.setdefault(INPUT_SMILES, str(in_smi or ""))
    props.setdefault(CANONICAL_SMILES, str(canonical or ""))
    props.setdefault(SMILES, str(canonical or ""))
    props.setdefault(INCHIKEY, str(key or ""))
    if source_format:
        props.setdefault(SOURCE_FORMAT, source_format)
    if row_index:
        props.setdefault(SOURCE_ROW_INDEX, int(row_index))
    props.setdefault(
        ROW_ID,
        _stable_row_id(
            source_format=final_source_format,
            row_index=final_row_index,
            mol_id=str(record_id),
            input_smiles=str(in_smi or ""),
        ),
    )
    return cm


def contract_record_from_chemmol(cm: ChemMol, row_index: int) -> MoleculeContractRecord:
    cm = ensure_contract_props(cm, row_index=row_index)
    props = cm.props or {}
    return MoleculeContractRecord(
        row_index=row_index,
        mol_id=str(props.get(MOL_ID, f"mol_{row_index}")),
        name="" if cm.name is None else str(cm.name),
        input_smiles=str(props.get(INPUT_SMILES, props.get(SMILES, ""))),
        canonical_smiles=str(props.get(CANONICAL_SMILES, props.get(SMILES, ""))),
        standardized_smiles=str(props.get(STANDARDIZED_SMILES, props.get("SMILES_STD", ""))),
        inchikey=str(props.get(INCHIKEY, "")),
        source_format=str(props.get(SOURCE_FORMAT, props.get("IMPORT_SOURCE_FORMAT", ""))),
        props=dict(props),
    )


def contract_issues_for_mols(mols: Iterable[ChemMol]) -> list[ServiceIssue]:
    issues: list[ServiceIssue] = []
    seen: dict[str, int] = {}
    for i, cm in enumerate(mols, start=1):
        props = cm.props or {}
        key = str(props.get(INCHIKEY) or "")
        if not props.get(SMILES) and not props.get(CANONICAL_SMILES):
            issues.append(ServiceIssue("MISSING_CANONICAL_SMILES", "Molecule has no canonical SMILES property.", "warning", row_index=i))
        if key:
            seen[key] = seen.get(key, 0) + 1
    for key, count in seen.items():
        if count > 1:
            issues.append(ServiceIssue("DUPLICATE_INCHIKEY", f"InChIKey occurs {count} times.", "warning", molecule_id=key))
    return issues
