from __future__ import annotations


class ChemCoreError(RuntimeError):
    """Base exception for chem_inf_widgets chemcore services."""


class MoleculeContractError(ChemCoreError):
    """Raised when an input table/list does not satisfy the molecule contract."""


class MoleculeParsingError(ChemCoreError):
    """Raised when molecule parsing fails in a non-recoverable context."""
