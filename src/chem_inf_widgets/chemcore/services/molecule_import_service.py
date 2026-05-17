from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - depends on runtime environment
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import (
    CANONICAL_SMILES,
    DROPPED_REASON,
    INCHIKEY,
    INPUT_SMILES,
    MOL_ID,
    QC_DUPLICATE_COUNT,
    QC_DUPLICATE_KEY,
    SOURCE_FORMAT,
    SOURCE_ROW_INDEX,
    append_qc_flag,
    append_transform_step,
    ensure_contract_props,
    set_dropped_reason,
)
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles, safe_mol_from_smiles

IMPORT_HUB_VERSION = "0.2.0"
SMILES_COLUMN_CANDIDATES = (
    "smiles",
    "canonical_smiles",
    "canonical smiles",
    "isomeric_smiles",
    "smile",
    "structure",
    "mol_smiles",
)
NAME_COLUMN_CANDIDATES = (
    "name",
    "title",
    "compound_name",
    "compound",
    "molecule_name",
    "molecule_id",
    "compound_id",
    "id",
    "identifier",
)
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".smi", ".smiles", ".sdf", ".sd"}


@dataclass(frozen=True)
class MoleculeImportConfig:
    """Configuration for Molecule Import Hub."""

    smiles_column: Optional[str] = None
    name_column: Optional[str] = None
    delimiter: Optional[str] = None
    sanitize: bool = True
    remove_hs: bool = True
    keep_failed_rows: bool = True
    max_preview_rows: int = 50
    flag_duplicates: bool = True
    reject_duplicate_structures: bool = False
    duplicate_key: str = "inchikey"  # inchikey or canonical_smiles


@dataclass(frozen=True)
class MoleculeImportRecord:
    row_index: int
    source_format: str
    source_name: str
    name: str
    input_smiles: str
    canonical_smiles: str
    ok: bool
    status: str
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    props: Dict[str, Any] = field(default_factory=dict)
    inchikey: str = ""
    mol_id: str = ""
    duplicate_key: str = ""
    duplicate_count: int = 1
    duplicate_group_index: int = 0
    accepted: bool = True
    rejection_reason: str = ""


@dataclass(frozen=True)
class MoleculeImportSummary:
    source_path: str
    source_format: str
    total_records: int
    valid_records: int
    failed_records: int
    accepted_records: int = 0
    rejected_records: int = 0
    duplicate_groups: int = 0
    duplicate_records: int = 0
    smiles_column: str = ""
    name_column: str = ""
    columns: List[str] = field(default_factory=list)
    version: str = IMPORT_HUB_VERSION


@dataclass(frozen=True)
class MoleculeImportResult:
    mols: List[ChemMol]
    records: List[MoleculeImportRecord]
    summary: MoleculeImportSummary

    @property
    def failed_records(self) -> List[MoleculeImportRecord]:
        return [r for r in self.records if not r.ok]

    @property
    def valid_records(self) -> List[MoleculeImportRecord]:
        return [r for r in self.records if r.ok]

    @property
    def accepted_records(self) -> List[MoleculeImportRecord]:
        return [r for r in self.records if r.accepted]

    @property
    def rejected_records(self) -> List[MoleculeImportRecord]:
        return [r for r in self.records if not r.accepted]


def _normalize_name(name: str) -> str:
    return str(name or "").strip().lower().replace("_", " ").replace("-", " ")


def _find_column(columns: Sequence[str], preferred: Optional[str], candidates: Sequence[str]) -> str:
    if preferred:
        for c in columns:
            if c.strip().lower() == preferred.strip().lower():
                return c
        raise ValueError(f"Column '{preferred}' was not found. Available columns: {', '.join(columns)}")
    normalized = {_normalize_name(c): c for c in columns}
    for cand in candidates:
        key = _normalize_name(cand)
        if key in normalized:
            return normalized[key]
    # exact lower-case fallback without whitespace normalization
    lower = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return ""


def _detect_delimiter(path: Path, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit == "\\t":
            return "\t"
        return explicit
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "\t"
    try:
        sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except Exception:
        return ","




def _safe_inchikey(mol: Any) -> str:
    if Chem is None or mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(Chem.Mol(mol)) or ""
    except Exception:
        return ""


def _record_with_updates(record: MoleculeImportRecord, **updates: Any) -> MoleculeImportRecord:
    values = asdict(record)
    values.update(updates)
    return MoleculeImportRecord(**values)


def _finalize_import_result(result: MoleculeImportResult, config: MoleculeImportConfig) -> MoleculeImportResult:
    """Annotate valid records with duplicate/acceptance information.

    The importer keeps backward-compatible behaviour by accepting duplicate
    structures unless ``reject_duplicate_structures`` is enabled. Failed rows are
    always rejected and remain available in the report output.
    """
    if not config.flag_duplicates:
        records = [
            _record_with_updates(r, accepted=bool(r.ok), rejection_reason="" if r.ok else (r.error or "failed"))
            for r in result.records
        ]
        summary = _summary_with_acceptance(result.summary, records)
        return MoleculeImportResult(mols=list(result.mols), records=records, summary=summary)

    keys: Dict[str, List[int]] = {}
    for idx, rec in enumerate(result.records):
        if not rec.ok:
            continue
        key = rec.inchikey if config.duplicate_key == "inchikey" else rec.canonical_smiles
        key = str(key or "")
        if key:
            keys.setdefault(key, []).append(idx)

    records: List[MoleculeImportRecord] = []
    for idx, rec in enumerate(result.records):
        if not rec.ok:
            records.append(_record_with_updates(rec, accepted=False, rejection_reason=rec.error or "failed"))
            continue

        key = rec.inchikey if config.duplicate_key == "inchikey" else rec.canonical_smiles
        group = keys.get(str(key or ""), [idx])
        duplicate_count = len(group)
        group_index = group.index(idx) + 1 if idx in group else 1
        warnings = list(rec.warnings or [])
        accepted = True
        rejection_reason = ""
        status = rec.status

        if duplicate_count > 1:
            message = f"Duplicate structure {group_index}/{duplicate_count} by {config.duplicate_key}."
            if message not in warnings:
                warnings.append(message)
            if config.reject_duplicate_structures and group_index > 1:
                accepted = False
                rejection_reason = "duplicate_structure"
                status = "duplicate_rejected"

        records.append(_record_with_updates(
            rec,
            warnings=warnings,
            duplicate_key=str(key or ""),
            duplicate_count=duplicate_count,
            duplicate_group_index=group_index,
            accepted=accepted,
            rejection_reason=rejection_reason,
            status=status,
        ))

    # Update ChemMol props in valid-record order so downstream widgets can audit
    # where each molecule came from and whether it was accepted by the gatekeeper.
    valid_records = [r for r in records if r.ok]
    for cm, rec in zip(result.mols, valid_records):
        props = cm.props if isinstance(cm.props, dict) else {}
        cm.props = props
        props["IMPORT_ACCEPTED"] = int(bool(rec.accepted))
        props["IMPORT_REJECTION_REASON"] = rec.rejection_reason
        props["IMPORT_DUPLICATE_KEY"] = rec.duplicate_key
        props["IMPORT_DUPLICATE_COUNT"] = int(rec.duplicate_count)
        props["IMPORT_DUPLICATE_GROUP_INDEX"] = int(rec.duplicate_group_index)
        props[QC_DUPLICATE_KEY] = rec.duplicate_key
        props[QC_DUPLICATE_COUNT] = int(rec.duplicate_count)
        if rec.duplicate_count > 1:
            append_qc_flag(cm, "duplicate_structure")
        if not rec.accepted:
            set_dropped_reason(cm, rec.rejection_reason or rec.error or "failed")
        elif not rec.rejection_reason and props.get(DROPPED_REASON):
            props[DROPPED_REASON] = ""
        if rec.warnings:
            props["IMPORT_WARNINGS"] = " | ".join(rec.warnings)

    summary = _summary_with_acceptance(result.summary, records)
    return MoleculeImportResult(mols=list(result.mols), records=records, summary=summary)


def _summary_with_acceptance(summary: MoleculeImportSummary, records: Sequence[MoleculeImportRecord]) -> MoleculeImportSummary:
    duplicate_groups = len({r.duplicate_key for r in records if r.ok and r.duplicate_key and r.duplicate_count > 1})
    duplicate_records = sum(1 for r in records if r.ok and r.duplicate_count > 1)
    return MoleculeImportSummary(
        source_path=summary.source_path,
        source_format=summary.source_format,
        total_records=summary.total_records,
        valid_records=summary.valid_records,
        failed_records=summary.failed_records,
        accepted_records=sum(1 for r in records if r.accepted),
        rejected_records=sum(1 for r in records if not r.accepted),
        duplicate_groups=duplicate_groups,
        duplicate_records=duplicate_records,
        smiles_column=summary.smiles_column,
        name_column=summary.name_column,
        columns=list(summary.columns),
        version=summary.version,
    )

def detect_import_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".sdf", ".sd"}:
        return "sdf"
    if suffix in {".smi", ".smiles"}:
        return "smi"
    if suffix == ".tsv":
        return "tsv"
    if suffix in {".csv", ".txt"}:
        return "table"
    raise ValueError(f"Unsupported file extension '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")


def _read_table_rows(path: Path, config: MoleculeImportConfig) -> Tuple[List[Dict[str, Any]], List[str], str]:
    delimiter = _detect_delimiter(path, config.delimiter)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, columns, delimiter


def _parse_smiles_record(
    *,
    row_index: int,
    source_format: str,
    source_name: str,
    smiles: str,
    name: str,
    props: Dict[str, Any],
    config: MoleculeImportConfig,
) -> Tuple[Optional[ChemMol], MoleculeImportRecord]:
    parsed = safe_mol_from_smiles(str(smiles or "").strip(), sanitize=bool(config.sanitize), remove_hs=bool(config.remove_hs))
    warnings = list(parsed.warnings or [])
    if not parsed.ok or parsed.mol is None:
        return None, MoleculeImportRecord(
            row_index=row_index,
            source_format=source_format,
            source_name=source_name,
            name=str(name or ""),
            input_smiles=str(smiles or ""),
            canonical_smiles="",
            ok=False,
            status="failed",
            error=parsed.error or "Invalid or empty SMILES.",
            warnings=warnings,
            props=props,
        )
    canonical = parsed.canonical_smiles or safe_canonical_smiles(parsed.mol, remove_hs=True)
    mol_props = dict(props)
    mol_props.setdefault("SMILES", canonical)
    mol_props.setdefault(INPUT_SMILES, str(smiles or ""))
    mol_props.setdefault(CANONICAL_SMILES, canonical)
    mol_props.setdefault(MOL_ID, str(name or f"mol_{row_index}"))
    mol_props[SOURCE_FORMAT] = source_format
    mol_props[SOURCE_ROW_INDEX] = row_index
    mol_props["IMPORT_SOURCE_FORMAT"] = source_format
    mol_props["IMPORT_ROW_INDEX"] = row_index
    if warnings:
        mol_props["IMPORT_WARNINGS"] = " | ".join(warnings)
    cm = ensure_contract_props(
        ChemMol(mol=parsed.mol, name=str(name or "") or None, props=mol_props, cache={}),
        row_index=row_index,
        source_format=source_format,
        input_smiles=str(smiles or ""),
    )
    append_transform_step(cm, f"import_{source_format}")
    inchikey = str(cm.props.get(INCHIKEY) or _safe_inchikey(parsed.mol))
    mol_id = str(cm.props.get(MOL_ID) or name or f"mol_{row_index}")
    return cm, MoleculeImportRecord(
        row_index=row_index,
        source_format=source_format,
        source_name=source_name,
        name=str(name or ""),
        input_smiles=str(smiles or ""),
        canonical_smiles=canonical,
        ok=True,
        status="imported",
        warnings=warnings,
        props=props,
        inchikey=inchikey,
        mol_id=mol_id,
        duplicate_key=inchikey or canonical,
        accepted=True,
    )


def import_table_file(path: str | Path, config: Optional[MoleculeImportConfig] = None) -> MoleculeImportResult:
    cfg = config or MoleculeImportConfig()
    p = Path(path)
    rows, columns, delimiter = _read_table_rows(p, cfg)
    smiles_col = _find_column(columns, cfg.smiles_column, SMILES_COLUMN_CANDIDATES)
    if not smiles_col:
        raise ValueError("No SMILES-like column was detected. Specify --smiles-column.")
    name_col = _find_column(columns, cfg.name_column, NAME_COLUMN_CANDIDATES) if columns else ""

    mols: List[ChemMol] = []
    records: List[MoleculeImportRecord] = []
    for i, row in enumerate(rows, start=1):
        smiles = row.get(smiles_col, "")
        name = row.get(name_col, "") if name_col else f"mol_{i}"
        props = {k: v for k, v in row.items() if k not in {smiles_col, name_col}}
        props["IMPORT_DELIMITER"] = "tab" if delimiter == "\t" else delimiter
        cm, rec = _parse_smiles_record(
            row_index=i,
            source_format="table",
            source_name=p.name,
            smiles=smiles,
            name=name,
            props=props,
            config=cfg,
        )
        if cm is not None:
            mols.append(cm)
        records.append(rec)
    summary = MoleculeImportSummary(
        source_path=str(p),
        source_format="table",
        total_records=len(records),
        valid_records=sum(1 for r in records if r.ok),
        failed_records=sum(1 for r in records if not r.ok),
        smiles_column=smiles_col,
        name_column=name_col,
        columns=columns,
    )
    return MoleculeImportResult(mols=mols, records=records, summary=summary)


def _split_smi_line(line: str) -> Tuple[str, str, Dict[str, Any]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return "", "", {}
    parts = line.split()
    smiles = parts[0]
    name = parts[1] if len(parts) > 1 else ""
    if len(parts) > 2:
        name = " ".join(parts[1:])
    return smiles, name, {}


def import_smi_file(path: str | Path, config: Optional[MoleculeImportConfig] = None) -> MoleculeImportResult:
    cfg = config or MoleculeImportConfig()
    p = Path(path)
    mols: List[ChemMol] = []
    records: List[MoleculeImportRecord] = []
    row_index = 0
    with p.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            row_index += 1
            smiles, name, props = _split_smi_line(raw_line)
            cm, rec = _parse_smiles_record(
                row_index=row_index,
                source_format="smi",
                source_name=p.name,
                smiles=smiles,
                name=name or f"mol_{row_index}",
                props=props,
                config=cfg,
            )
            if cm is not None:
                mols.append(cm)
            records.append(rec)
    summary = MoleculeImportSummary(
        source_path=str(p),
        source_format="smi",
        total_records=len(records),
        valid_records=sum(1 for r in records if r.ok),
        failed_records=sum(1 for r in records if not r.ok),
        smiles_column="first_token",
        name_column="remaining_tokens",
        columns=["smiles", "name"],
    )
    return MoleculeImportResult(mols=mols, records=records, summary=summary)


def import_sdf_file(path: str | Path, config: Optional[MoleculeImportConfig] = None) -> MoleculeImportResult:
    if Chem is None:
        raise ImportError("RDKit is required for SDF import.")
    cfg = config or MoleculeImportConfig()
    p = Path(path)
    supplier = Chem.SDMolSupplier(str(p), sanitize=bool(cfg.sanitize), removeHs=bool(cfg.remove_hs))
    mols: List[ChemMol] = []
    records: List[MoleculeImportRecord] = []
    columns: set[str] = set()
    for i, mol in enumerate(supplier, start=1):
        if mol is None:
            records.append(MoleculeImportRecord(
                row_index=i,
                source_format="sdf",
                source_name=p.name,
                name=f"mol_{i}",
                input_smiles="",
                canonical_smiles="",
                ok=False,
                status="failed",
                error="RDKit could not parse this SDF record.",
            ))
            continue
        props = {k: mol.GetProp(k) for k in mol.GetPropNames()} if hasattr(mol, "GetPropNames") else {}
        columns.update(props.keys())
        name = ""
        try:
            name = mol.GetProp("_Name")
        except Exception:
            name = props.get("name") or props.get("Name") or f"mol_{i}"
        canonical = safe_canonical_smiles(mol, remove_hs=True)
        props.setdefault("SMILES", canonical)
        props.setdefault(INPUT_SMILES, canonical)
        props.setdefault(CANONICAL_SMILES, canonical)
        props.setdefault(MOL_ID, name or f"mol_{i}")
        props[SOURCE_FORMAT] = "sdf"
        props[SOURCE_ROW_INDEX] = i
        props["IMPORT_SOURCE_FORMAT"] = "sdf"
        props["IMPORT_ROW_INDEX"] = i
        cm = ensure_contract_props(
            ChemMol(mol=mol, name=name or None, props=props, cache={}),
            row_index=i,
            source_format="sdf",
            input_smiles=canonical,
        )
        append_transform_step(cm, "import_sdf")
        inchikey = str(cm.props.get(INCHIKEY) or _safe_inchikey(mol))
        mol_id = str(cm.props.get(MOL_ID) or name or f"mol_{i}")
        mols.append(cm)
        records.append(MoleculeImportRecord(
            row_index=i,
            source_format="sdf",
            source_name=p.name,
            name=name or f"mol_{i}",
            input_smiles=canonical,
            canonical_smiles=canonical,
            ok=True,
            status="imported",
            props={k: v for k, v in props.items() if k not in {"SMILES", "IMPORT_SOURCE_FORMAT", "IMPORT_ROW_INDEX"}},
            inchikey=inchikey,
            mol_id=mol_id,
            duplicate_key=inchikey or canonical,
            accepted=True,
        ))
    summary = MoleculeImportSummary(
        source_path=str(p),
        source_format="sdf",
        total_records=len(records),
        valid_records=sum(1 for r in records if r.ok),
        failed_records=sum(1 for r in records if not r.ok),
        smiles_column="SDF structure",
        name_column="_Name",
        columns=sorted(columns),
    )
    return MoleculeImportResult(mols=mols, records=records, summary=summary)


def import_molecule_file(path: str | Path, config: Optional[MoleculeImportConfig] = None) -> MoleculeImportResult:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    fmt = detect_import_format(p)
    cfg = config or MoleculeImportConfig()
    if fmt == "sdf":
        result = import_sdf_file(p, cfg)
    elif fmt == "smi":
        result = import_smi_file(p, cfg)
    else:
        result = import_table_file(p, cfg)
    return _finalize_import_result(result, cfg)


def import_records_as_dicts(records: Iterable[MoleculeImportRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        qc_flags: List[str] = []
        if rec.duplicate_count > 1:
            qc_flags.append("duplicate_structure")
        if not rec.ok:
            qc_flags.append("invalid_structure")
        dropped_reason = rec.rejection_reason or ""
        if not rec.ok:
            dropped_reason = "invalid_structure"
        rows.append({
            "row_index": rec.row_index,
            "source_format": rec.source_format,
            "source_name": rec.source_name,
            "name": rec.name,
            "input_smiles": rec.input_smiles,
            "canonical_smiles": rec.canonical_smiles,
            "inchikey": rec.inchikey,
            "mol_id": rec.mol_id,
            "ok": int(bool(rec.ok)),
            "accepted": int(bool(rec.accepted)),
            "status": rec.status,
            "error": rec.error,
            "warnings": " | ".join(rec.warnings or []),
            "duplicate_key": rec.duplicate_key,
            "duplicate_count": rec.duplicate_count,
            "duplicate_group_index": rec.duplicate_group_index,
            "rejection_reason": rec.rejection_reason,
            "qc_flags": " | ".join(qc_flags),
            "dropped_reason": dropped_reason,
            "props_json": json.dumps(rec.props, ensure_ascii=False, sort_keys=True),
        })
    return rows


def import_summary_as_rows(summary: MoleculeImportSummary) -> List[Dict[str, Any]]:
    return [
        {"metric": "source_path", "value": summary.source_path, "description": "Input file path."},
        {"metric": "source_format", "value": summary.source_format, "description": "Detected input format."},
        {"metric": "total_records", "value": summary.total_records, "description": "All input records."},
        {"metric": "valid_records", "value": summary.valid_records, "description": "Successfully imported molecules."},
        {"metric": "failed_records", "value": summary.failed_records, "description": "Records that failed parsing/import."},
        {"metric": "accepted_records", "value": summary.accepted_records, "description": "Records accepted by the import gatekeeper."},
        {"metric": "rejected_records", "value": summary.rejected_records, "description": "Records rejected by the import gatekeeper."},
        {"metric": "duplicate_groups", "value": summary.duplicate_groups, "description": "Duplicate structure groups detected."},
        {"metric": "duplicate_records", "value": summary.duplicate_records, "description": "Valid records belonging to duplicate groups."},
        {"metric": "smiles_column", "value": summary.smiles_column, "description": "SMILES source column or equivalent."},
        {"metric": "name_column", "value": summary.name_column, "description": "Name/identifier source column or equivalent."},
        {"metric": "columns", "value": ";".join(summary.columns), "description": "Input columns/properties detected."},
        {"metric": "import_hub_version", "value": summary.version, "description": "Import service version."},
    ]
