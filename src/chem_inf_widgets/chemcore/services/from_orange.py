from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table, Variable

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import append_transform_step, ensure_contract_props, is_meta_field
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles
from chem_inf_widgets.chemcore.services.orange_table_utils import as_float_or_nan, column_is_numeric, looks_like_meta_key, records_to_orange_table

from ..models.chembl_dataset import ChemBLDataset


try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


SMILES_CANDIDATES = {"smiles", "canonical_smiles", "smile"}


@dataclass(frozen=True)
class TableMolConversionReport:
    n_rows: int
    n_valid: int
    n_invalid: int
    skipped_rows: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    smiles_column: Optional[str] = None
    name_column: Optional[str] = None


def _is_string_var(v: Variable) -> bool:
    return isinstance(v, StringVariable)


def _iter_all_vars(domain: Domain) -> List[Variable]:
    return list(domain.metas) + list(domain.attributes) + list(domain.class_vars)


def _find_var_by_name_ci(domain: Domain, name: str) -> Optional[Variable]:
    name_l = name.strip().lower()
    for v in _iter_all_vars(domain):
        if v.name.strip().lower() == name_l:
            return v
    return None


def _auto_smiles_var(domain: Domain) -> Optional[StringVariable]:
    # Prefer well-known names
    for v in _iter_all_vars(domain):
        if _is_string_var(v) and v.name.strip().lower() in SMILES_CANDIDATES:
            return v  # type: ignore[return-value]

    # Fallback: first string variable
    for v in _iter_all_vars(domain):
        if _is_string_var(v):
            return v  # type: ignore[return-value]
    return None


def _col_as_str_list(data: Table, var: Variable) -> List[str]:
    col = data.get_column(var)
    out: List[str] = []
    for x in col:
        if x is None:
            out.append("")
        else:
            out.append(str(x))
    return out


def _unique_name(existing: set[str], name: str) -> str:
    if name not in existing:
        existing.add(name)
        return name
    i = 2
    while f"{name}_{i}" in existing:
        i += 1
    nn = f"{name}_{i}"
    existing.add(nn)
    return nn


def _cell_to_python_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if isinstance(value, (np.generic,)):
            return value.item()
    except Exception:
        pass

    text = str(value)
    if text == "?":
        return None
    return value


def table_to_chemmols_with_report(
    data: Table,
    *,
    smiles_var: Optional[str] = None,
    name_var: Optional[str] = None,
    prop_keys: Optional[Sequence[str]] = None,
    sanitize: bool = True,
) -> tuple[List[ChemMol], TableMolConversionReport]:
    """
    Convert Orange Table -> list[ChemMol] using a SMILES column and return
    a structured conversion report.
    """
    if Chem is None:
        raise ImportError("RDKit is required to convert SMILES to molecules.")

    if data is None:
        return [], TableMolConversionReport(0, 0, 0, [], [])

    domain = data.domain

    if smiles_var:
        smiles_v = _find_var_by_name_ci(domain, smiles_var)
    else:
        smiles_v = _auto_smiles_var(domain)

    if smiles_v is None or not _is_string_var(smiles_v):
        raise ValueError("No SMILES column found in the input Table.")

    smiles_list = _col_as_str_list(data, smiles_v)

    name_v: Optional[Variable] = None
    if name_var:
        name_v = _find_var_by_name_ci(domain, name_var)
    else:
        for cand in ("name", "title", "compound", "compound_name", "id", "ime", "ime spojine"):
            vv = _find_var_by_name_ci(domain, cand)
            if vv is not None and vv != smiles_v:
                name_v = vv
                break

    name_list: Optional[List[str]] = _col_as_str_list(data, name_v) if name_v is not None else None

    all_vars = _iter_all_vars(domain)
    skip = {smiles_v.name}
    if name_v is not None:
        skip.add(name_v.name)

    if prop_keys is None:
        prop_vars = [v for v in all_vars if v.name not in skip]
    else:
        wanted = set(prop_keys)
        prop_vars = [v for v in all_vars if v.name in wanted and v.name not in skip]

    prop_cols: Dict[str, np.ndarray] = {}
    for v in prop_vars:
        col = data.get_column(v)
        prop_cols[v.name] = np.asarray(col, dtype=object)

    out: List[ChemMol] = []
    skipped_rows: List[int] = []
    errors: List[str] = []

    for i, smi in enumerate(smiles_list):
        parse_result = safe_mol_from_smiles(smi, sanitize=bool(sanitize), remove_hs=True)
        if not parse_result.ok or parse_result.mol is None:
            skipped_rows.append(i + 1)
            if parse_result.error:
                errors.append(f"Row {i + 1}: {parse_result.error}")
            continue

        props: Dict[str, Any] = {"SMILES": parse_result.canonical_smiles or "", "input_smiles": str(smi or ""), "canonical_smiles": parse_result.canonical_smiles or ""}
        for k, col in prop_cols.items():
            value = _cell_to_python_value(col[i])
            if value is None:
                continue
            props[k] = value

        if parse_result.warnings:
            props["PARSE_WARNINGS"] = " | ".join(parse_result.warnings)

        nm = name_list[i] if name_list is not None else None
        nm = (nm or "").strip() if isinstance(nm, str) else nm

        cm = ChemMol(
            mol=parse_result.mol,
            name=nm or None,
            props=props,
            cache={},
        )
        cm = ensure_contract_props(cm, row_index=i + 1, input_smiles=str(smi or ""))
        append_transform_step(cm, "table_to_chemmols")
        out.append(cm)

    report = TableMolConversionReport(
        n_rows=len(data),
        n_valid=len(out),
        n_invalid=len(skipped_rows),
        skipped_rows=skipped_rows,
        errors=errors,
        smiles_column=smiles_v.name,
        name_column=name_v.name if name_v is not None else None,
    )
    return out, report


def table_to_chemmols(
    data: Table,
    *,
    smiles_var: Optional[str] = None,
    name_var: Optional[str] = None,
    prop_keys: Optional[Sequence[str]] = None,
    sanitize: bool = True,
) -> List[ChemMol]:
    """
    Convert Orange Table -> list[ChemMol] using a SMILES column.

    - smiles_var: explicit name of SMILES column; if None, auto-detect.
    - name_var: optional title column; if None, tries common names ("name", "title", ...).
    - prop_keys: list of property columns to copy into ChemMol.props.
                 If None -> copy all except smiles/name.
    """
    mols, _report = table_to_chemmols_with_report(
        data,
        smiles_var=smiles_var,
        name_var=name_var,
        prop_keys=prop_keys,
        sanitize=sanitize,
    )
    return mols


def chemmols_to_table(
    mols: list[ChemMol],
    prop_keys: list[str] | None = None,
) -> Table:
    """Convert ChemMol objects to an Orange Table.

    Structural identifiers and provenance fields are exported as metas, while
    numeric molecule properties are exported as continuous attributes. This is
    important for QSAR workflows: an imported decimal activity column must be a
    real Orange variable, not only a meta column.
    """

    # Fill the shared molecule contract first, then collect property keys and values.
    keys = set()
    for i, cm in enumerate(mols, start=1):
        ensure_contract_props(cm, row_index=i)
        keys.update(cm.props.keys())

    if prop_keys:
        keys &= set(prop_keys)

    keys.discard("SMILES")
    keys.discard("Name")

    prop_values = {key: [cm.props.get(key, "") for cm in mols] for key in sorted(keys)}
    attribute_keys = [
        key
        for key, values in prop_values.items()
        if not looks_like_meta_key(key) and not is_meta_field(key) and column_is_numeric(values)
    ]
    meta_prop_keys = [key for key in sorted(keys) if key not in set(attribute_keys)]

    attributes = [ContinuousVariable(key) for key in attribute_keys]
    metas = [StringVariable("SMILES"), StringVariable("Name")] + [StringVariable(key) for key in meta_prop_keys]
    domain = Domain(attributes, [], metas)

    X = np.asarray(
        [[as_float_or_nan(cm.props.get(key, "")) for key in attribute_keys] for cm in mols],
        dtype=float,
    ) if mols else np.empty((0, len(attribute_keys)), dtype=float)

    M = np.asarray(
        [
            [
                cm.props.get("SMILES", ""),
                "" if cm.name is None else str(cm.name),
                *["" if cm.props.get(key, "") is None else str(cm.props.get(key, "")) for key in meta_prop_keys],
            ]
            for cm in mols
        ],
        dtype=object,
    ) if mols else np.empty((0, len(metas)), dtype=object)

    return Table.from_numpy(domain, X=X, metas=M)

def dataset_to_table(ds: ChemBLDataset) -> Table:
    """Convert a ChemBLDataset to an Orange Table using shared role inference.

    Numeric bioactivity/property columns now become Orange attributes instead of
    being hidden among metas. Identifier, structure, unit/relation and provenance
    columns remain metas.
    """
    table = records_to_orange_table(ds.props, name="ChemBL Dataset")
    if table is not None:
        return table
    return Table.from_numpy(Domain([], metas=[]), X=np.empty((0, 0), dtype=float), metas=np.empty((0, 0), dtype=object))
