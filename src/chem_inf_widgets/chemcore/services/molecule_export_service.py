from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from Orange.data import Table, Variable

from chem_inf_widgets.chemcore.io.sdf import write_sdf
from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import CANONICAL_SMILES, INCHIKEY, INPUT_SMILES, MOL_ID, ensure_contract_props
from chem_inf_widgets.chemcore.services.from_orange import TableMolConversionReport, table_to_chemmols_with_report

EXPORT_HUB_VERSION = "0.1.0"
SUPPORTED_EXPORT_EXTENSIONS = {".csv", ".tsv", ".txt", ".smi", ".smiles", ".sdf", ".sd"}
STRUCTURE_COLUMN_NAME = "SMILES"
NAME_COLUMN_NAME = "Name"


@dataclass(frozen=True)
class MoleculeExportConfig:
    """Configuration for Molecule Export Hub."""

    output_format: Optional[str] = None
    smiles_column: Optional[str] = None
    name_column: Optional[str] = None
    delimiter: Optional[str] = None
    sanitize: bool = True
    include_props: bool = True
    write_name: bool = True
    include_header: bool = True
    use_canonical_smiles: bool = True


@dataclass(frozen=True)
class MoleculeExportRecord:
    row_index: int
    source_kind: str
    source_name: str
    name: str
    input_smiles: str
    canonical_smiles: str
    output_smiles: str
    ok: bool
    written: bool
    status: str
    error: str = ""
    mol_id: str = ""
    inchikey: str = ""
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MoleculeExportSummary:
    output_path: str
    output_format: str
    source_kind: str
    source_name: str
    total_records: int
    valid_records: int
    failed_records: int
    written_records: int
    skipped_records: int
    smiles_column: str = ""
    name_column: str = ""
    delimiter: str = ""
    columns: List[str] = field(default_factory=list)
    version: str = EXPORT_HUB_VERSION


@dataclass(frozen=True)
class MoleculeExportResult:
    records: List[MoleculeExportRecord]
    summary: MoleculeExportSummary

    @property
    def failed_records(self) -> List[MoleculeExportRecord]:
        return [record for record in self.records if not record.ok or not record.written]


@dataclass
class _PreparedExport:
    source_kind: str
    source_name: str
    records: List[MoleculeExportRecord]
    valid_molecules: List[ChemMol]
    smiles_column: str = ""
    name_column: str = ""


def _normalize_format_name(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "auto"}:
        return ""
    if text in {"csv", "tsv", "txt", "sdf"}:
        return text
    if text in {"sd"}:
        return "sdf"
    if text in {"smi", "smiles"}:
        return "smi"
    raise ValueError(f"Unsupported export format '{value}'.")


def detect_export_format(path: str | Path, explicit: Optional[str] = None) -> str:
    normalized = _normalize_format_name(explicit)
    if normalized:
        return normalized

    suffix = Path(path).suffix.lower()
    if suffix in {".sdf", ".sd"}:
        return "sdf"
    if suffix in {".smi", ".smiles"}:
        return "smi"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".csv":
        return "csv"
    if suffix == ".txt":
        return "txt"
    raise ValueError(f"Unsupported file extension '{suffix}'. Supported: {sorted(SUPPORTED_EXPORT_EXTENSIONS)}")


def _record_with_updates(record: MoleculeExportRecord, **updates: Any) -> MoleculeExportRecord:
    values = asdict(record)
    values.update(updates)
    return MoleculeExportRecord(**values)


def _is_nan(value: Any) -> bool:
    try:
        return isinstance(value, float) and math.isnan(value)
    except Exception:
        return False


def _orange_value_to_python(var: Variable, value: Any) -> Any:
    if _is_nan(value):
        return ""
    if hasattr(var, "values") and getattr(var, "values", None) and isinstance(value, (int, float, np.floating)):
        try:
            idx = int(value)
            if 0 <= idx < len(var.values):
                return var.values[idx]
        except Exception:
            pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _table_column_map(data: Table) -> dict[str, list[Any]]:
    variables = list(data.domain.attributes) + list(data.domain.class_vars) + list(data.domain.metas)
    return {
        var.name: [_orange_value_to_python(var, value) for value in data.get_column(var)]
        for var in variables
    }


def _table_input_lists(
    data: Table,
    report: TableMolConversionReport,
) -> tuple[list[Any], list[Any]]:
    columns = _table_column_map(data)
    smiles_values = columns.get(report.smiles_column or "", [])
    name_values = columns.get(report.name_column or "", []) if report.name_column else []
    if not name_values:
        name_values = [f"mol_{idx}" for idx in range(1, len(data) + 1)]
    return list(smiles_values), list(name_values)


def _copy_and_contract(cm: ChemMol, *, row_index: int, source_kind: str) -> ChemMol:
    copied = cm.copy()
    ensure_contract_props(copied, row_index=row_index, source_format=source_kind)
    return copied


def _record_from_molecule(
    cm: ChemMol,
    *,
    row_index: int,
    source_kind: str,
    source_name: str,
    config: MoleculeExportConfig,
) -> MoleculeExportRecord:
    props = dict(cm.props or {})
    canonical_smiles = str(props.get(CANONICAL_SMILES) or cm.canonical_smiles(remove_hs=True) or "")
    input_smiles = str(props.get(INPUT_SMILES) or props.get(STRUCTURE_COLUMN_NAME) or canonical_smiles or "")
    output_smiles = canonical_smiles if config.use_canonical_smiles else (input_smiles or canonical_smiles)
    mol_id = str(props.get(MOL_ID) or cm.name or f"mol_{row_index}")
    name = str(cm.name or props.get(NAME_COLUMN_NAME) or mol_id)
    inchikey = str(props.get(INCHIKEY) or "")
    return MoleculeExportRecord(
        row_index=row_index,
        source_kind=source_kind,
        source_name=source_name,
        name=name,
        input_smiles=input_smiles,
        canonical_smiles=canonical_smiles,
        output_smiles=output_smiles,
        ok=bool(cm.mol is not None),
        written=False,
        status="ready",
        error="",
        mol_id=mol_id,
        inchikey=inchikey,
        props=props,
    )


def _prepare_from_molecules(
    molecules: Sequence[ChemMol],
    config: MoleculeExportConfig,
) -> _PreparedExport:
    records: List[MoleculeExportRecord] = []
    valid_molecules: List[ChemMol] = []
    source_name = "Molecules input"
    for row_index, molecule in enumerate(molecules, start=1):
        if molecule is None or molecule.mol is None:
            records.append(MoleculeExportRecord(
                row_index=row_index,
                source_kind="molecules",
                source_name=source_name,
                name=f"mol_{row_index}",
                input_smiles="",
                canonical_smiles="",
                output_smiles="",
                ok=False,
                written=False,
                status="failed",
                error="Missing RDKit molecule.",
            ))
            continue
        prepared = _copy_and_contract(molecule, row_index=row_index, source_kind="molecules")
        valid_molecules.append(prepared)
        records.append(_record_from_molecule(
            prepared,
            row_index=row_index,
            source_kind="molecules",
            source_name=source_name,
            config=config,
        ))
    return _PreparedExport(
        source_kind="molecules",
        source_name=source_name,
        records=records,
        valid_molecules=valid_molecules,
    )


def _prepare_from_table(data: Table, config: MoleculeExportConfig) -> _PreparedExport:
    valid_molecules, report = table_to_chemmols_with_report(
        data,
        smiles_var=config.smiles_column or None,
        name_var=config.name_column or None,
        sanitize=bool(config.sanitize),
    )
    source_name = str(data.name or "Data")
    smiles_values, name_values = _table_input_lists(data, report)
    skipped_rows = list(report.skipped_rows or [])
    skipped_set = set(skipped_rows)
    error_map = {
        row_index: (
            report.errors[idx] if idx < len(report.errors) and report.errors[idx] else "Invalid or empty SMILES."
        )
        for idx, row_index in enumerate(skipped_rows)
    }

    records: List[MoleculeExportRecord] = []
    prepared_valid_molecules: List[ChemMol] = []
    valid_iter = iter(valid_molecules)

    for row_index in range(1, len(data) + 1):
        if row_index in skipped_set:
            records.append(MoleculeExportRecord(
                row_index=row_index,
                source_kind="data",
                source_name=source_name,
                name=str(name_values[row_index - 1] or f"mol_{row_index}"),
                input_smiles=str(smiles_values[row_index - 1] or ""),
                canonical_smiles="",
                output_smiles="",
                ok=False,
                written=False,
                status="failed",
                error=str(error_map.get(row_index, "Invalid or empty SMILES.")),
            ))
            continue

        prepared = next(valid_iter)
        prepared_valid_molecules.append(prepared)
        records.append(_record_from_molecule(
            prepared,
            row_index=row_index,
            source_kind="data",
            source_name=source_name,
            config=config,
        ))

    return _PreparedExport(
        source_kind="data",
        source_name=source_name,
        records=records,
        valid_molecules=prepared_valid_molecules,
        smiles_column=report.smiles_column or "",
        name_column=report.name_column or "",
    )


def _normalize_delimiter(output_format: str, explicit: Optional[str]) -> str:
    if explicit:
        return "\t" if explicit == "\\t" else explicit
    if output_format == "csv":
        return ","
    return "\t"


def _normalize_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if _is_nan(value):
        return ""
    return value


def _extra_property_keys(records: Sequence[MoleculeExportRecord]) -> list[str]:
    excluded = {
        STRUCTURE_COLUMN_NAME.lower(),
        NAME_COLUMN_NAME.lower(),
        INPUT_SMILES.lower(),
        CANONICAL_SMILES.lower(),
    }
    keys = set()
    for record in records:
        if not record.ok:
            continue
        keys.update(
            key for key in (record.props or {}).keys()
            if str(key).strip().lower() not in excluded
        )
    return sorted(str(key) for key in keys)


def _delimited_fieldnames(records: Sequence[MoleculeExportRecord], config: MoleculeExportConfig) -> list[str]:
    fieldnames = [STRUCTURE_COLUMN_NAME]
    if config.write_name:
        fieldnames.append(NAME_COLUMN_NAME)
    if config.include_props:
        fieldnames.extend(_extra_property_keys(records))
    return fieldnames


def _record_to_row(record: MoleculeExportRecord, fieldnames: Sequence[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for fieldname in fieldnames:
        if fieldname == STRUCTURE_COLUMN_NAME:
            row[fieldname] = record.output_smiles
        elif fieldname == NAME_COLUMN_NAME:
            row[fieldname] = record.name
        else:
            row[fieldname] = _normalize_cell_value((record.props or {}).get(fieldname, ""))
    return row


def _write_delimited_file(
    path: Path,
    records: Sequence[MoleculeExportRecord],
    *,
    delimiter: str,
    config: MoleculeExportConfig,
) -> list[str]:
    fieldnames = _delimited_fieldnames(records, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        if config.include_header:
            writer.writeheader()
        for record in records:
            if not record.ok or not record.output_smiles:
                continue
            writer.writerow(_record_to_row(record, fieldnames))
    return fieldnames


def _write_smiles_file(
    path: Path,
    records: Sequence[MoleculeExportRecord],
    *,
    config: MoleculeExportConfig,
) -> list[str]:
    columns = [STRUCTURE_COLUMN_NAME]
    if config.write_name:
        columns.append(NAME_COLUMN_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if config.include_header:
            handle.write("\t".join(columns) + "\n")
        for record in records:
            if not record.ok or not record.output_smiles:
                continue
            parts = [record.output_smiles]
            if config.write_name:
                parts.append(record.name)
            handle.write("\t".join(parts).rstrip() + "\n")
    return columns


def _sdf_property_keys(records: Sequence[MoleculeExportRecord], config: MoleculeExportConfig) -> list[str]:
    if not config.include_props:
        return []
    return _extra_property_keys(records)


def _write_sdf_file(
    path: Path,
    records: Sequence[MoleculeExportRecord],
    valid_molecules: Sequence[ChemMol],
    *,
    config: MoleculeExportConfig,
) -> list[str]:
    prop_keys = _sdf_property_keys(records, config)
    include_props: bool | Iterable[str]
    if not config.include_props:
        include_props = False
    elif prop_keys:
        include_props = prop_keys
    else:
        include_props = True

    write_sdf(
        [molecule.copy() for molecule in valid_molecules],
        path,
        include_props=include_props,
        write_name=bool(config.write_name),
    )
    columns: list[str] = []
    if config.write_name:
        columns.append("_Name")
    if config.include_props:
        columns.extend(prop_keys)
    return columns


def _mark_written_records(
    records: Sequence[MoleculeExportRecord],
    *,
    output_format: str,
) -> list[MoleculeExportRecord]:
    written_records: list[MoleculeExportRecord] = []
    for record in records:
        if not record.ok:
            written_records.append(record)
            continue
        if output_format != "sdf" and not record.output_smiles:
            written_records.append(_record_with_updates(
                record,
                written=False,
                status="failed",
                error="No SMILES value available for export.",
                ok=False,
            ))
            continue
        written_records.append(_record_with_updates(record, written=True, status="written"))
    return written_records


def export_molecule_data(
    path: str | Path,
    *,
    data: Optional[Table] = None,
    molecules: Optional[Sequence[ChemMol]] = None,
    config: Optional[MoleculeExportConfig] = None,
) -> MoleculeExportResult:
    cfg = config or MoleculeExportConfig()
    output_path = Path(path)
    output_format = detect_export_format(output_path, cfg.output_format)

    if molecules is not None:
        prepared = _prepare_from_molecules(molecules, cfg)
    elif data is not None:
        prepared = _prepare_from_table(data, cfg)
    else:
        raise ValueError("No input data or molecules provided for export.")

    if output_format in {"csv", "tsv", "txt"}:
        columns = _write_delimited_file(
            output_path,
            prepared.records,
            delimiter=_normalize_delimiter(output_format, cfg.delimiter),
            config=cfg,
        )
    elif output_format == "smi":
        columns = _write_smiles_file(output_path, prepared.records, config=cfg)
    elif output_format == "sdf":
        columns = _write_sdf_file(output_path, prepared.records, prepared.valid_molecules, config=cfg)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported export format '{output_format}'.")

    records = _mark_written_records(prepared.records, output_format=output_format)
    summary = MoleculeExportSummary(
        output_path=str(output_path),
        output_format=output_format,
        source_kind=prepared.source_kind,
        source_name=prepared.source_name,
        total_records=len(records),
        valid_records=sum(1 for record in records if record.ok),
        failed_records=sum(1 for record in records if not record.ok),
        written_records=sum(1 for record in records if record.written),
        skipped_records=sum(1 for record in records if not record.written),
        smiles_column=prepared.smiles_column,
        name_column=prepared.name_column,
        delimiter=_normalize_delimiter(output_format, cfg.delimiter) if output_format in {"csv", "tsv", "txt"} else "",
        columns=list(columns),
    )
    return MoleculeExportResult(records=records, summary=summary)


def export_records_as_dicts(records: Iterable[MoleculeExportRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append({
            "row_index": record.row_index,
            "source_kind": record.source_kind,
            "source_name": record.source_name,
            "name": record.name,
            "mol_id": record.mol_id,
            "input_smiles": record.input_smiles,
            "canonical_smiles": record.canonical_smiles,
            "output_smiles": record.output_smiles,
            "inchikey": record.inchikey,
            "ok": int(bool(record.ok)),
            "written": int(bool(record.written)),
            "status": record.status,
            "error": record.error,
            "props_json": json.dumps(record.props, ensure_ascii=False, sort_keys=True, default=str),
        })
    return rows


def export_summary_as_rows(summary: MoleculeExportSummary) -> List[Dict[str, Any]]:
    return [
        {"metric": "output_path", "value": summary.output_path, "description": "Export file path."},
        {"metric": "output_format", "value": summary.output_format, "description": "Selected export format."},
        {"metric": "source_kind", "value": summary.source_kind, "description": "Active widget input used for export."},
        {"metric": "source_name", "value": summary.source_name, "description": "Input table name or source label."},
        {"metric": "total_records", "value": summary.total_records, "description": "All input records seen by the export hub."},
        {"metric": "valid_records", "value": summary.valid_records, "description": "Records converted into exportable molecules."},
        {"metric": "failed_records", "value": summary.failed_records, "description": "Records that failed conversion before export."},
        {"metric": "written_records", "value": summary.written_records, "description": "Records written to disk."},
        {"metric": "skipped_records", "value": summary.skipped_records, "description": "Records not written to disk."},
        {"metric": "smiles_column", "value": summary.smiles_column, "description": "Detected SMILES source column for Table input."},
        {"metric": "name_column", "value": summary.name_column, "description": "Detected name source column for Table input."},
        {"metric": "delimiter", "value": summary.delimiter, "description": "Delimiter used for delimited text export."},
        {"metric": "columns", "value": ";".join(summary.columns), "description": "Columns or properties written to the export file."},
        {"metric": "export_hub_version", "value": summary.version, "description": "Export service version."},
    ]
