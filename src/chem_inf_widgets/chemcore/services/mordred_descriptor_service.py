from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem

try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ModuleNotFoundError:
    Calculator = None
    descriptors = None
    MORDRED_AVAILABLE = False

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


@dataclass(frozen=True)
class MordredComputeConfig:
    ignore_3d: bool = True
    nproc: Optional[int] = 1  # safe default: single process in Orange GUI


@dataclass(frozen=True)
class DescriptorInfo:
    name: str
    module: str


class MordredDescriptorService:
    """
    Chemcore service for:
      - listing available Mordred descriptors (2D/3D switch)
      - computing selected descriptors for RDKit mols
    """

    def __init__(self, cfg: Optional[MordredComputeConfig] = None) -> None:
        if not MORDRED_AVAILABLE:
            raise ImportError("MordredDescriptorService requires the optional 'mordred' package.")
        self.cfg = cfg or MordredComputeConfig()

        # Build global descriptor list once
        calc_all = Calculator(descriptors, ignore_3D=self.cfg.ignore_3d)
        self._all_desc = list(calc_all.descriptors)

        # Map "descriptor_name" -> (descriptor_object, module)
        self._desc_map: Dict[str, Tuple[object, str]] = {}
        for d in self._all_desc:
            self._desc_map[str(d)] = (d, getattr(d, "__module__", ""))

    def list_descriptors(self) -> List[DescriptorInfo]:
        out: List[DescriptorInfo] = []
        for name, (_obj, mod) in self._desc_map.items():
            out.append(DescriptorInfo(name=name, module=mod))
        out.sort(key=lambda x: (x.module, x.name))
        return out

    def compute(
        self,
        mols: Sequence[Chem.Mol],
        selected_descriptor_names: Sequence[str],
        *,
        cfg: Optional[MordredComputeConfig] = None,
    ) -> pd.DataFrame:
        """
        Compute selected descriptors for RDKit mols.

        Returns:
          pandas DataFrame with one row per mol in input order.
          If mordred returns non-numeric objects, we coerce to numeric where possible,
          otherwise NaN for those entries.
        """
        if not mols:
            return pd.DataFrame()

        names = [n for n in selected_descriptor_names if n in self._desc_map]
        if not names:
            return pd.DataFrame(index=range(len(mols)))

        use_cfg = cfg or self.cfg

        calc = Calculator([], ignore_3D=use_cfg.ignore_3d)
        for n in names:
            calc.register(self._desc_map[n][0])

        # Mordred uses tqdm by default. In an Orange/PyQt background worker,
        # tqdm can trigger repeated QSocketNotifier warnings and can destabilize
        # the GUI. Run quietly and suppress only progress/descriptor warnings.
        nproc = 1 if use_cfg.nproc in (None, 0) else int(use_cfg.nproc)
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            try:
                df = calc.pandas(list(mols), nproc=nproc, quiet=True)
            except TypeError:
                # Older Mordred versions do not expose ``quiet``. stderr is still
                # redirected above, so tqdm/progress output remains disabled.
                df = calc.pandas(list(mols), nproc=nproc)

        # Convert to numeric where possible; keep NaN for failures
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df[names]

    @staticmethod
    def smiles_to_mols(smiles: Sequence[str]) -> Tuple[List[Optional[Chem.Mol]], List[int]]:
        """
        Parse SMILES in order.

        Returns:
          mols_maybe: list same length as smiles (None for invalid)
          valid_idx: indices of valid mols
        """
        mols_maybe: List[Optional[Chem.Mol]] = []
        valid_idx: List[int] = []
        for i, smi in enumerate(smiles):
            s = (smi or "").strip()
            mol = safe_mol_from_smiles(s, sanitize=True, remove_hs=True).mol if s else None
            mols_maybe.append(mol)
            if mol is not None:
                valid_idx.append(i)
        return mols_maybe, valid_idx

    @staticmethod
    def chemmols_to_mols(molecules: Sequence[ChemMol]) -> Tuple[List[Optional[Chem.Mol]], List[int]]:
        """
        Convert ChemMol objects into RDKit mols while preserving row alignment.

        Falls back to the ``SMILES`` property when the ChemMol instance does not
        currently hold an RDKit molecule.
        """
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
        """
        Expand df of valid mols into full length df (invalid rows = NaN).
        """
        if df_valid.empty:
            return pd.DataFrame(index=range(n_total))
        df_valid = df_valid.copy()
        df_valid.index = valid_idx
        return df_valid.reindex(range(n_total))
