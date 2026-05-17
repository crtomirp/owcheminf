from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles

from .chembl_models import ChemBLBioactivityRecord, ChemBLMoleculeRecord
from .chembl_molecule_service import ChemBLMoleculePropsRecord

if TYPE_CHECKING:
    from Orange.data import Table


def _safe_chemmol_from_smiles(smiles: str, *, name: str) -> Optional[ChemMol]:
    parsed = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True)
    if parsed.mol is None:
        return None
    return ChemMol.from_rdkit(parsed.mol, name=name)


def _to_float_nan(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def derive_prop_keys_from_records(
    props_by_id: Dict[str, ChemBLMoleculePropsRecord],
    max_keys: int = 25,
) -> List[str]:
    if not props_by_id:
        return []

    keys = set()
    for record in props_by_id.values():
        for key in (record.props or {}).keys():
            if key:
                keys.add(str(key))

    if not keys:
        return []

    preferred = ["full_mwt", "alogp", "psa", "hba", "hbd", "rtb", "qed_weighted"]
    ordered = [key for key in preferred if key in keys]
    if len(ordered) < max_keys:
        rest = sorted(key for key in keys if key not in ordered)
        ordered.extend(rest[: max(0, max_keys - len(ordered))])
    return ordered[:max_keys]


def _bio_value(record: ChemBLBioactivityRecord, key: str) -> Any:
    if key == "pChEMBL":
        return getattr(record, "pchembl_value", None)
    if key == "standard_value":
        return getattr(record, "standard_value", None)
    if key == "IC50_nM":
        return getattr(record, "ic50_nM", None)
    if key == "standard_type":
        return getattr(record, "standard_type", "") or ""
    if key == "standard_units":
        return getattr(record, "standard_units", "") or ""
    if key == "assay_chembl_id":
        return getattr(record, "assay_chembl_id", "") or ""
    if key == "target_chembl_id":
        return getattr(record, "target_chembl_id", "") or ""
    if key == "molecule_chembl_id":
        return getattr(record, "molecule_chembl_id", "") or ""
    if key == "SMILES":
        return getattr(record, "smiles", "") or ""
    if key == "pref_name":
        return ""
    return getattr(record, key, None)


def _mode(values: Iterable[str]) -> str:
    filtered = [value for value in values if value]
    if not filtered:
        return ""
    counts = Counter(filtered)
    return counts.most_common(1)[0][0]


def split_bio_fields(
    selected_bio_fields: Sequence[str],
    bio_field_specs: Sequence[Tuple[str, str]],
) -> Tuple[List[str], List[str]]:
    kind_map = {key: kind for key, kind in bio_field_specs}
    numeric = [key for key in selected_bio_fields if kind_map.get(key) == "num"]
    metadata = [
        key
        for key in selected_bio_fields
        if kind_map.get(key) in ("meta", "smiles")
        and key not in ("SMILES", "molecule_chembl_id", "pref_name")
    ]
    return numeric, metadata


def aggregate_bio_by_molecule(
    recs: Sequence[ChemBLBioactivityRecord],
    selected_bio_fields: Sequence[str],
    bio_field_specs: Sequence[Tuple[str, str]],
) -> Dict[str, Dict[str, Any]]:
    num_fields, meta_fields = split_bio_fields(selected_bio_fields, bio_field_specs)

    buckets: Dict[str, List[ChemBLBioactivityRecord]] = defaultdict(list)
    for record in recs or []:
        molecule_id = (getattr(record, "molecule_chembl_id", "") or "").strip().upper()
        if molecule_id:
            buckets[molecule_id].append(record)

    out: Dict[str, Dict[str, Any]] = {}
    for molecule_id, records in buckets.items():
        aggregated: Dict[str, Any] = {"bio_n": len(records)}

        for key in num_fields:
            values: List[float] = []
            for record in records:
                value = _bio_value(record, key)
                try:
                    if value is None or value == "":
                        continue
                    values.append(float(value))
                except Exception:
                    continue

            if not values:
                aggregated[key] = None
            elif key == "pChEMBL":
                aggregated[key] = max(values)
            elif key in ("standard_value", "IC50_nM"):
                aggregated[key] = min(values)
            else:
                aggregated[key] = max(values)

        for key in meta_fields:
            values = [str(_bio_value(record, key) or "").strip() for record in records]
            aggregated[key] = _mode(values)

        out[molecule_id] = aggregated

    return out


def _normalize_props_by_id(
    props_by_id: Dict[str, ChemBLMoleculePropsRecord],
) -> Dict[str, ChemBLMoleculePropsRecord]:
    if not props_by_id:
        return {}
    return {str(key).strip().upper(): value for key, value in props_by_id.items()}


def build_bioactivity_outputs(
    recs: Sequence[ChemBLBioactivityRecord],
    prop_keys: Sequence[str],
    props_by_id: Dict[str, ChemBLMoleculePropsRecord],
    selected_bio_fields: Sequence[str],
    bio_field_specs: Sequence[Tuple[str, str]],
    include_props_in_molecules: bool = True,
) -> Tuple[Optional[Table], List[ChemMol]]:
    from Orange.data import ContinuousVariable, Domain, StringVariable, Table

    if not recs:
        return None, []

    prop_keys = list(prop_keys or [])
    normalized_props = _normalize_props_by_id(props_by_id)
    bio_by_id = aggregate_bio_by_molecule(recs, selected_bio_fields, bio_field_specs)

    attrs = [ContinuousVariable(key) for key in prop_keys]
    attrs.extend([ContinuousVariable("standard_value"), ContinuousVariable("pChEMBL")])

    smiles_var = StringVariable("SMILES")
    smiles_var.attributes["format"] = "SMILES"
    metas = [
        StringVariable("molecule_chembl_id"),
        StringVariable("assay_chembl_id"),
        StringVariable("target_chembl_id"),
        StringVariable("standard_type"),
        StringVariable("standard_units"),
        smiles_var,
    ]
    domain = Domain(attrs, metas=metas)

    rows_x: List[List[float]] = []
    rows_m: List[List[object]] = []
    out_mols: List[ChemMol] = []

    for record in recs:
        molecule_id_raw = str(getattr(record, "molecule_chembl_id", "") or "")
        molecule_id = molecule_id_raw.strip().upper()
        prop_record = normalized_props.get(molecule_id)

        smiles = str(getattr(record, "smiles", "") or "")
        standard_type = str(getattr(record, "standard_type", "") or "")
        standard_units = str(getattr(record, "standard_units", "") or "")
        assay_id = str(getattr(record, "assay_chembl_id", "") or "")
        target_id = str(getattr(record, "target_chembl_id", "") or "")

        numeric_row = [
            _to_float_nan((prop_record.props or {}).get(key) if prop_record is not None else None)
            for key in prop_keys
        ]
        numeric_row.append(_to_float_nan(getattr(record, "standard_value", None)))
        numeric_row.append(_to_float_nan(getattr(record, "pchembl_value", None)))
        rows_x.append(numeric_row)
        rows_m.append([molecule_id_raw, assay_id, target_id, standard_type, standard_units, smiles])

        if not smiles:
            continue

        chem_mol = _safe_chemmol_from_smiles(smiles, name=molecule_id_raw)
        if chem_mol is None:
            continue
        chem_mol.set_prop("molecule_chembl_id", molecule_id_raw)
        chem_mol.set_prop("assay_chembl_id", assay_id)
        chem_mol.set_prop("target_chembl_id", target_id)
        chem_mol.set_prop("standard_type", standard_type)
        chem_mol.set_prop("standard_units", standard_units)
        chem_mol.set_prop("standard_value", getattr(record, "standard_value", None))
        chem_mol.set_prop("pChEMBL", getattr(record, "pchembl_value", None))

        if include_props_in_molecules and prop_record is not None:
            for key in prop_keys:
                if key in (prop_record.props or {}):
                    chem_mol.set_prop(key, prop_record.props.get(key))

        bio = bio_by_id.get(molecule_id, {})
        if bio:
            chem_mol.set_prop("bio_n", bio.get("bio_n", 0))
            for key, kind in bio_field_specs:
                if kind == "num":
                    chem_mol.set_prop(key, bio.get(key, None))
                elif kind in ("meta", "smiles"):
                    chem_mol.set_prop(key, bio.get(key, ""))

        out_mols.append(chem_mol)

    x_np = np.array(rows_x, dtype=float) if attrs else np.zeros((len(rows_m), 0), dtype=float)
    m_np = np.array(rows_m, dtype=object) if metas else np.zeros((len(rows_m), 0), dtype=object)
    return Table.from_numpy(domain, X=x_np, metas=m_np), out_mols


def build_molecule_outputs(
    mols: Sequence[ChemBLMoleculeRecord],
    props_by_id: Dict[str, ChemBLMoleculePropsRecord],
    prop_keys: Sequence[str],
    recs: Sequence[ChemBLBioactivityRecord],
    selected_bio_fields: Sequence[str],
    bio_field_specs: Sequence[Tuple[str, str]],
    include_props_in_molecules: bool = True,
    selected_target_id: Optional[str] = None,
) -> Tuple[Optional[Table], List[ChemMol]]:
    from Orange.data import ContinuousVariable, Domain, StringVariable, Table

    if not mols:
        return None, []

    prop_keys = list(prop_keys or [])
    normalized_props = _normalize_props_by_id(props_by_id)
    bio_by_id = aggregate_bio_by_molecule(recs, selected_bio_fields, bio_field_specs) if recs else {}
    add_bio = bool(bio_by_id)
    bio_num_fields, bio_meta_fields = split_bio_fields(selected_bio_fields, bio_field_specs)

    attrs = [ContinuousVariable(key) for key in prop_keys]
    if add_bio:
        attrs.append(ContinuousVariable("bio_n"))
        for key in bio_num_fields:
            if key not in prop_keys:
                attrs.append(ContinuousVariable(key))

    smiles_var = StringVariable("SMILES")
    smiles_var.attributes["format"] = "SMILES"
    metas: List[StringVariable] = [StringVariable("ChEMBL ID"), StringVariable("Name"), smiles_var]
    if add_bio:
        for key in bio_meta_fields:
            metas.append(StringVariable(key))
    domain = Domain(attrs, metas=metas)

    rows_x: List[List[float]] = []
    rows_m: List[List[object]] = []
    out_mols: List[ChemMol] = []

    for molecule in mols:
        molecule_id_raw = (molecule.chembl_id or "").strip()
        if not molecule_id_raw:
            continue
        molecule_id = molecule_id_raw.upper()

        prop_record = normalized_props.get(molecule_id)
        name = (prop_record.pref_name if prop_record else "") or (molecule.pref_name or "")
        smiles = (prop_record.canonical_smiles if prop_record else "") or (molecule.canonical_smiles or "")
        bio = bio_by_id.get(molecule_id, {})

        numeric_row = [
            _to_float_nan((prop_record.props or {}).get(key) if prop_record is not None else None)
            for key in prop_keys
        ]
        if add_bio:
            numeric_row.append(_to_float_nan(bio.get("bio_n", None)))
            for key in bio_num_fields:
                if key in prop_keys:
                    continue
                numeric_row.append(_to_float_nan(bio.get(key, None)))
        rows_x.append(numeric_row)

        meta_row: List[object] = [molecule_id_raw, name, smiles]
        if add_bio:
            for key in bio_meta_fields:
                meta_row.append(str(bio.get(key, "") or ""))
        rows_m.append(meta_row)

        if not smiles:
            continue

        chem_mol = _safe_chemmol_from_smiles(smiles, name=molecule_id_raw)
        if chem_mol is None:
            continue
        chem_mol.set_prop("molecule_chembl_id", molecule_id_raw)
        chem_mol.set_prop("pref_name", name)
        if selected_target_id:
            chem_mol.set_prop("target_chembl_id", selected_target_id)

        if include_props_in_molecules and prop_record is not None:
            for key in prop_keys:
                if key in (prop_record.props or {}):
                    chem_mol.set_prop(key, prop_record.props.get(key))

        if add_bio and bio:
            chem_mol.set_prop("bio_n", bio.get("bio_n", 0))
            for key in bio_num_fields:
                chem_mol.set_prop(key, bio.get(key, None))
            for key in bio_meta_fields:
                chem_mol.set_prop(key, bio.get(key, ""))

        out_mols.append(chem_mol)

    x_np = np.array(rows_x, dtype=float) if attrs else np.zeros((len(rows_m), 0), dtype=float)
    m_np = np.array(rows_m, dtype=object) if metas else np.zeros((len(rows_m), 0), dtype=object)
    return Table.from_numpy(domain, X=x_np, metas=m_np), out_mols
