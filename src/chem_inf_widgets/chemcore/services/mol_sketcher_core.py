from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


# ------------------------- optional RDKit -------------------------

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    Descriptors = None
    Lipinski = None
    RDKIT_AVAILABLE = False


# ------------------------- config models -------------------------

@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    type: str = "string"   # "string" | "float" | "int"


@dataclass(frozen=True)
class MetadataSpec:
    name: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class DbKeySpec:
    name: str
    label: str
    type: str = "int"
    initial: int = 1


@dataclass
class SketcherConfig:
    json_path: str = ""
    fields: List[FieldSpec] = field(default_factory=list)
    user_metadata: List[MetadataSpec] = field(default_factory=list)
    dbkey: Optional[DbKeySpec] = None


# ------------------------- RDKit property map -------------------------

def _property_map() -> Dict[str, Tuple[str, Callable]]:
    """
    Map config field 'name' -> (output label (unused here), calculator)
    We use FieldSpec.label for the actual output column name.
    """
    if not RDKIT_AVAILABLE:
        return {}

    return {
        "mw": ("Molecular Weight", Descriptors.MolWt),
        "logp": ("LogP", Descriptors.MolLogP),
        "tpsa": ("TPSA", Descriptors.TPSA),
        "hbd": ("H-Bond Donors", Lipinski.NumHDonors),
        "hba": ("H-Bond Acceptors", Lipinski.NumHAcceptors),
        "rotatable_bonds": ("Rotatable Bonds", Lipinski.NumRotatableBonds),
        "inchi": ("InChI", Chem.MolToInchi),
        "inchikey": ("InChI Key", Chem.MolToInchiKey),
    }


# ------------------------- public API -------------------------

class MolSketcherCore:
    """
    Chemcore logic for JSME sketcher widget:
      - load JSON config
      - compute RDKit properties for SMILES
      - maintain DB key counter
      - build Orange Table
      - (optional) build ChemMol list
    """

    def __init__(self) -> None:
        self.config = SketcherConfig()
        self._dbkey_counter: int = 1
        self._rows: List[Dict[str, Any]] = []

    # ---------- config ----------

    def load_config(self, json_path: str) -> SketcherConfig:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        dbkey = None
        if isinstance(raw.get("dbkey"), dict):
            d = raw["dbkey"]
            dbkey = DbKeySpec(
                name=str(d.get("name", "dbkey")),
                label=str(d.get("label", "DBKEY")),
                type=str(d.get("type", "int")),
                initial=int(d.get("initial", 1)),
            )

        fields_raw = raw.get("fields", []) or []
        fields: List[FieldSpec] = []
        for it in fields_raw:
            fields.append(
                FieldSpec(
                    name=str(it.get("name", "")),
                    label=str(it.get("label", it.get("name", ""))),
                    type=str(it.get("type", "string")),
                )
            )

        # If dbkey exists and not already present in fields -> prepend
        if dbkey is not None and not any(f.name == dbkey.name for f in fields):
            fields = [FieldSpec(name=dbkey.name, label=dbkey.label, type=dbkey.type)] + fields

        meta_raw = raw.get("user_metadata", []) or []
        user_metadata: List[MetadataSpec] = []
        for it in meta_raw:
            user_metadata.append(
                MetadataSpec(
                    name=str(it.get("name", "")),
                    label=str(it.get("label", it.get("name", ""))),
                    description=str(it.get("description", "")),
                )
            )

        self.config = SketcherConfig(
            json_path=json_path,
            fields=fields,
            user_metadata=user_metadata,
            dbkey=dbkey,
        )

        self._dbkey_counter = dbkey.initial if dbkey is not None else 1
        return self.config

    # ---------- data manipulation ----------

    def clear(self) -> None:
        self._rows.clear()
        self._dbkey_counter = self.config.dbkey.initial if self.config.dbkey is not None else 1

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return self._rows

    def add_compound(self, smiles: str, metadata_values: Dict[str, str]) -> Dict[str, Any]:
        """
        Create one compound row dict and append into internal buffer.
        Returns the created row.
        """
        smi = (smiles or "").strip()
        if not smi:
            raise ValueError("Empty SMILES")

        row: Dict[str, Any] = {}

        # Always include smiles
        row["smiles"] = smi

        # DBKEY if configured
        if self.config.dbkey is not None:
            row[self.config.dbkey.label] = int(self._dbkey_counter)
            self._dbkey_counter += 1

        # RDKit props (if available and requested in fields)
        if RDKIT_AVAILABLE:
            mol = safe_mol_from_smiles(smi, sanitize=True, remove_hs=True).mol
            if mol is None:
                raise ValueError("Invalid SMILES structure")

            pmap = _property_map()
            for f in self.config.fields:
                # dbkey handled above; smiles handled below
                if f.name == "smiles":
                    row[f.label] = smi
                    continue
                if self.config.dbkey is not None and f.name == self.config.dbkey.name:
                    # already emitted under dbkey.label
                    continue

                if f.name in pmap:
                    _lbl, calc = pmap[f.name]
                    try:
                        row[f.label] = calc(mol)
                    except Exception:
                        row[f.label] = np.nan

        # user metadata (stored under their labels)
        for meta in self.config.user_metadata:
            row[meta.label] = str(metadata_values.get(meta.name, ""))

        self._rows.append(row)
        return row

    # ---------- outputs ----------

    def build_table(self) -> Optional[Table]:
        """
        Build an Orange Table from internal rows.
        Numeric fields -> attributes; everything else -> metas.
        """
        if not self._rows:
            return None

        fields = self.config.fields or [FieldSpec("smiles", "smiles", "string")]
        metas_conf = self.config.user_metadata or []

        # Create variables
        attrs: List[ContinuousVariable] = []
        metas: List[StringVariable] = []

        # config fields first
        for f in fields:
            label = f.label
            if f.type in ("float", "int"):
                attrs.append(ContinuousVariable(label))
            else:
                metas.append(StringVariable(label))

        # then metadata fields
        for m in metas_conf:
            metas.append(StringVariable(m.label))

        # Ensure unique names inside domain
        taken = set()
        for v in list(attrs) + list(metas):
            if v.name in taken:
                # make unique
                base = v.name
                k = 2
                while f"{base}_{k}" in taken:
                    k += 1
                v.name = f"{base}_{k}"
            taken.add(v.name)

        domain = Domain(attrs, metas=metas)

        X: List[List[float]] = []
        M: List[List[object]] = []

        for comp in self._rows:
            x_row = [comp.get(var.name, np.nan) for var in attrs]
            m_row = [str(comp.get(var.name, "")) for var in metas]
            X.append(x_row)
            M.append(m_row)

        X_arr = np.array(X, dtype=float) if attrs else np.empty((len(X), 0), dtype=float)
        M_arr = np.array(M, dtype=object) if metas else np.empty((len(X), 0), dtype=object)

        table = Table.from_numpy(domain, X=X_arr, metas=M_arr)
        table.name = "Compounds"
        return table

    def build_molecules(self) -> List[ChemMol]:
        """
        Optional: export ChemMol list from current rows.
        Preserves row dict as props where possible.
        """
        out: List[ChemMol] = []
        if not self._rows:
            return out

        for i, row in enumerate(self._rows):
            smi = str(row.get("smiles", "")).strip()
            if not smi:
                continue

            cm: Optional[ChemMol] = None
            if RDKIT_AVAILABLE:
                mol = safe_mol_from_smiles(smi, sanitize=True, remove_hs=True).mol
                if mol is not None:
                    cm = ChemMol.from_rdkit(mol, name=row.get("Name", None) or f"mol_{i+1}")

            if cm is None:
                continue

            # push props
            for k, v in row.items():
                try:
                    cm.set_prop(str(k), v)
                except Exception:
                    pass

            out.append(cm)

        return out

    # ---------- resource path ----------

    @staticmethod
    def get_jsme_html_path() -> str:
        """
        Return absolute path to JSME HTML panel.
        You should place the JSME assets under:
          chem_inf_widgets/chemcore/resources/jsme/jsme_panel.html
        """
        here = os.path.dirname(__file__)
        # services/ -> chemcore/
        chemcore_dir = os.path.abspath(os.path.join(here, ".."))
        html_path = os.path.join(chemcore_dir, "resources", "jsme", "jsme_panel.html")
        return html_path
