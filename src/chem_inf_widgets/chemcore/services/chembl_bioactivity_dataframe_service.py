from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


NUMERIC_OUTPUT_COLUMNS = [
    "pchembl_value",
    "IC50_nM",
    "hbd",
    "hba",
    "rotable_bonds",
    "mw",
    "tpsa",
    "logp",
    "lipinski_deviations",
]

META_OUTPUT_COLUMNS = [
    "SMILES",
    "molecule_chembl_id",
    "target_chembl_id",
    "assay_chembl_id",
    "document_chembl_id",
    "target_organism",
    "target_name",
]


def fetch_bioactivity_dataframe(
    target_id: str,
    *,
    standard_type: str = "IC50",
    limit: int = 1000,
    timeout: int = 30,
) -> pd.DataFrame:
    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        "target_chembl_id": target_id,
        "standard_type": standard_type,
        "limit": limit,
    }
    data = []

    while url:
        response = requests.get(
            url,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data.extend(payload.get("activities", []))
        url = payload.get("page_meta", {}).get("next")
        params = None

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "pchembl_value" in df.columns:
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        df = df.dropna(subset=["pchembl_value"])
    return df


def convert_activity_to_nm(value: object, unit: object) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return float("nan")

    normalized_unit = str(unit or "").strip().lower()
    conversions: Dict[str, float] = {
        "m": 1e9,
        "µm": 1e3,
        "um": 1e3,
        "nm": 1.0,
        "nmol/l": 1.0,
        "pm": 1e-3,
    }
    factor = conversions.get(normalized_unit)
    if factor is None:
        return float("nan")
    return numeric_value * factor


def process_ic50_values(df: pd.DataFrame) -> pd.DataFrame:
    if "standard_value" not in df.columns or "standard_units" not in df.columns:
        return df

    out = df.copy()
    out["IC50_nM"] = out.apply(
        lambda row: convert_activity_to_nm(row.get("standard_value"), row.get("standard_units")),
        axis=1,
    )
    return out.drop(columns=["standard_value", "standard_units"])


def normalize_smiles_column(df: pd.DataFrame) -> pd.DataFrame:
    if "canonical_smiles" not in df.columns:
        return df
    return df.rename(columns={"canonical_smiles": "SMILES"})


def calculate_drug_properties(df: pd.DataFrame) -> pd.DataFrame:
    if "SMILES" not in df.columns:
        return df

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except ImportError:
        return df

    out = df.copy()
    prop_columns = ["hbd", "hba", "rotable_bonds", "mw", "tpsa", "logp", "lipinski_deviations"]
    for col in prop_columns:
        out[col] = np.nan

    for idx, smiles in out["SMILES"].fillna("").items():
        if not smiles:
            continue
        mol = safe_mol_from_smiles(str(smiles), sanitize=True, remove_hs=True).mol
        if mol is None:
            continue

        out.at[idx, "hbd"] = Lipinski.NumHDonors(mol)
        out.at[idx, "hba"] = Lipinski.NumHAcceptors(mol)
        out.at[idx, "rotable_bonds"] = Descriptors.NumRotatableBonds(mol)
        out.at[idx, "mw"] = Descriptors.MolWt(mol)
        out.at[idx, "tpsa"] = Descriptors.TPSA(mol)
        out.at[idx, "logp"] = Descriptors.MolLogP(mol)

        violations = 0
        violations += 1 if out.at[idx, "mw"] > 500 else 0
        violations += 1 if out.at[idx, "logp"] > 5 else 0
        violations += 1 if out.at[idx, "hbd"] > 5 else 0
        violations += 1 if out.at[idx, "hba"] > 10 else 0
        out.at[idx, "lipinski_deviations"] = violations

    for col in prop_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def filter_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing_num = [col for col in NUMERIC_OUTPUT_COLUMNS if col in df.columns]
    existing_meta = [col for col in META_OUTPUT_COLUMNS if col in df.columns]
    return df[existing_num + existing_meta]
