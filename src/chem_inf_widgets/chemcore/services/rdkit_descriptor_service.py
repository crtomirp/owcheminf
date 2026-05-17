from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


@dataclass(frozen=True)
class RdkitDescriptorInfo:
    name: str
    category: str
    description: str = ""


@dataclass(frozen=True)
class RdkitDescriptorPreset:
    key: str
    label: str
    description: str
    categories: Tuple[str, ...]
    descriptors: Tuple[str, ...] = ()


RDKIT_DESCRIPTOR_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "Constitutional / counts": (
        "HeavyAtomCount", "NHOHCount", "NOCount", "NumHAcceptors", "NumHDonors",
        "NumHeteroatoms", "NumRotatableBonds", "NumValenceElectrons", "RingCount",
        "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
        "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
        "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
        "FractionCSP3",
    ),
    "Physicochemical / drug-like": (
        "MolWt", "ExactMolWt", "HeavyAtomMolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA",
        "qed", "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
    ),
    "Lipinski / EState": (
        "NumHAcceptors", "NumHDonors", "NumRotatableBonds", "TPSA", "MolLogP", "MolWt",
        "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex",
    ),
    "Topology / connectivity": (
        "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
        "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha",
        "Kappa1", "Kappa2", "Kappa3", "Ipc",
    ),
    "BCUT / charge-related": (
        "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI",
        "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
    ),
    "VSA / surface area": (
        "PEOE_VSA", "SMR_VSA", "SlogP_VSA", "EState_VSA", "VSA_EState",
    ),
    "Fragments / functional groups": (
        "fr_", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
    ),
}

# Compact, useful default for QSAR teaching and first-pass modeling.
DEFAULT_RDKIT_QSAR = (
    "MolWt", "ExactMolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA", "qed",
    "HeavyAtomCount", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3", "BertzCT", "BalabanJ",
    "MaxEStateIndex", "MinEStateIndex", "BCUT2D_MWHI", "BCUT2D_MWLOW",
)

RDKIT_DESCRIPTOR_PRESETS: Tuple[RdkitDescriptorPreset, ...] = (
    RdkitDescriptorPreset(
        key="recommended_qsar",
        label="Recommended RDKit QSAR core",
        description="Small robust subset for fast QSAR/QSPR workflows: size, polarity, lipophilicity, rings, topology, and selected EState/BCUT descriptors.",
        categories=tuple(RDKIT_DESCRIPTOR_CATEGORIES.keys()),
        descriptors=DEFAULT_RDKIT_QSAR,
    ),
    RdkitDescriptorPreset(
        key="physchem_druglike",
        label="Descriptor family: physicochemical / drug-like",
        description="Molecular weight, logP, TPSA, molar refractivity, QED, charge, and surface-area descriptors.",
        categories=("Physicochemical / drug-like", "Lipinski / EState"),
    ),
    RdkitDescriptorPreset(
        key="constitutional_counts",
        label="Descriptor family: constitutional and counts",
        description="Atom, heteroatom, ring, donor/acceptor, rotatable-bond, and saturation/aromaticity counts.",
        categories=("Constitutional / counts",),
    ),
    RdkitDescriptorPreset(
        key="topology_connectivity",
        label="Descriptor family: topology and connectivity",
        description="Balaban, Bertz, Chi, Kappa, Hall-Kier, and related graph/topology descriptors.",
        categories=("Topology / connectivity",),
    ),
    RdkitDescriptorPreset(
        key="vsa_bcut_estate",
        label="Descriptor family: VSA / BCUT / EState",
        description="VSA families, BCUT descriptors, and EState descriptors useful for compact QSAR feature sets.",
        categories=("VSA / surface area", "BCUT / charge-related", "Lipinski / EState"),
    ),
    RdkitDescriptorPreset(
        key="fragments",
        label="Descriptor family: fragments / functional groups",
        description="RDKit fragment counters and Morgan density descriptors.",
        categories=("Fragments / functional groups",),
    ),
    RdkitDescriptorPreset(
        key="custom",
        label="Custom / manual category selection",
        description="Choose descriptor categories manually, then select individual descriptors.",
        categories=tuple(RDKIT_DESCRIPTOR_CATEGORIES.keys()),
    ),
    RdkitDescriptorPreset(
        key="all",
        label="All RDKit descriptors",
        description="Expose the full RDKit descriptor catalog available in the installed RDKit version.",
        categories=tuple(RDKIT_DESCRIPTOR_CATEGORIES.keys()),
    ),
)


def _matches_token(name: str, token: str) -> bool:
    if token.endswith("_"):
        return name.startswith(token)
    return name == token or name.startswith(token + "_")


class RdkitDescriptorService:
    """List and compute descriptors from ``rdkit.Chem.Descriptors``."""

    def __init__(self) -> None:
        self._functions: Dict[str, object] = {name: func for name, func in Descriptors._descList}
        self._category_map = self._build_category_map()

    def _build_category_map(self) -> Dict[str, str]:
        category_map: Dict[str, str] = {}
        for category, tokens in RDKIT_DESCRIPTOR_CATEGORIES.items():
            for name in self._functions:
                if any(_matches_token(name, token) for token in tokens):
                    category_map.setdefault(name, category)
        for name in self._functions:
            category_map.setdefault(name, "Other RDKit descriptors")
        return category_map

    def list_descriptors(self) -> List[RdkitDescriptorInfo]:
        return [
            RdkitDescriptorInfo(name=name, category=self._category_map.get(name, "Other RDKit descriptors"))
            for name in sorted(self._functions)
        ]

    def descriptor_names_for_categories(self, categories: Sequence[str]) -> List[str]:
        categories_set = set(categories)
        return [info.name for info in self.list_descriptors() if info.category in categories_set]

    def compute(self, mols: Sequence[Chem.Mol], selected_descriptor_names: Sequence[str]) -> pd.DataFrame:
        selected = [name for name in selected_descriptor_names if name in self._functions]
        rows: List[Dict[str, float]] = []
        for mol in mols:
            row: Dict[str, float] = {}
            for name in selected:
                func = self._functions[name]
                try:
                    value = func(mol)
                    row[name] = float(value) if value is not None else np.nan
                except Exception:
                    row[name] = np.nan
            rows.append(row)
        return pd.DataFrame(rows, columns=selected)

    @staticmethod
    def smiles_to_mols(smiles: Sequence[str]) -> Tuple[List[Optional[Chem.Mol]], List[int]]:
        mols_maybe: List[Optional[Chem.Mol]] = []
        valid_idx: List[int] = []
        for i, smi in enumerate(smiles):
            text = (smi or "").strip()
            mol = safe_mol_from_smiles(text, sanitize=True, remove_hs=True).mol if text else None
            mols_maybe.append(mol)
            if mol is not None:
                valid_idx.append(i)
        return mols_maybe, valid_idx

    @staticmethod
    def chemmols_to_mols(molecules: Sequence[ChemMol]) -> Tuple[List[Optional[Chem.Mol]], List[int]]:
        mols_maybe: List[Optional[Chem.Mol]] = []
        valid_idx: List[int] = []
        for i, chem_mol in enumerate(molecules):
            rdmol: Optional[Chem.Mol] = None
            if chem_mol is not None and hasattr(chem_mol, "to_rdkit"):
                rdmol = chem_mol.to_rdkit()
            if rdmol is None and chem_mol is not None:
                smiles = str(chem_mol.get_prop("SMILES") or "").strip()
                rdmol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol if smiles else None
            mols_maybe.append(rdmol)
            if rdmol is not None:
                valid_idx.append(i)
        return mols_maybe, valid_idx

    @staticmethod
    def numeric_or_none(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (float, int, np.floating, np.integer)):
            try:
                if np.isnan(value):
                    return None
            except TypeError:
                pass
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def df_to_full_length(df_valid: pd.DataFrame, valid_idx: List[int], n_total: int) -> pd.DataFrame:
        if df_valid.empty:
            return pd.DataFrame(index=range(n_total))
        df_valid = df_valid.copy()
        df_valid.index = valid_idx
        return df_valid.reindex(range(n_total))
