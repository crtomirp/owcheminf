from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rdkit import Chem

from chem_inf_widgets.chemcore.services.rdkit_safe import (
    safe_canonical_smiles,
    safe_mol_from_smiles,
)


@dataclass
class ChemMol:
    """
    Canonical molecule object shared across chemcore modules.
    """

    mol: Chem.Mol
    name: Optional[str] = None

    # semantic properties (ADMET flags, descriptors, QSAR outputs, ...)
    props: Dict[str, Any] = field(default_factory=dict)

    # optional cache for intermediate results
    cache: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_smiles(cls, smiles: str, name: Optional[str] = None) -> "ChemMol":
        parsed = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True)
        if parsed.mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return cls(mol=parsed.mol, name=name)

    @classmethod
    def from_rdkit(cls, mol: Chem.Mol, name: Optional[str] = None) -> "ChemMol":
        if mol is None:
            raise ValueError("RDKit Mol is None")
        return cls(mol=mol, name=name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def smiles(self) -> str:
        return safe_canonical_smiles(self.mol, remove_hs=False)


    def copy(self) -> "ChemMol":
        """Return a defensive copy of the molecule wrapper."""
        mol_copy = Chem.Mol(self.mol) if self.mol is not None else self.mol
        return ChemMol(
            mol=mol_copy,
            name=self.name,
            props=dict(self.props or {}),
            cache=dict(self.cache or {}),
        )

    def to_rdkit(self) -> Optional[Chem.Mol]:
        if self.mol is None:
            return None
        return Chem.Mol(self.mol)

    def set_prop(self, key: str, value: Any) -> None:
        self.props[key] = value

    def get_prop(self, key: str, default: Any = None) -> Any:
        return self.props.get(key, default)

    def has_prop(self, key: str) -> bool:
        return key in self.props
        
    def canonical_smiles(
        self,
        remove_hs: bool = True,
        canonical: bool = True,
        isomeric: bool = True,
    ) -> str:
        mol = self.mol
        if mol is None:
            return ""

        if remove_hs:
            try:
                mol = Chem.RemoveHs(mol)
            except Exception:
                pass

        return safe_canonical_smiles(
            mol,
            remove_hs=False,
            canonical=canonical,
            isomeric=isomeric,
        )
