"""Compatibility wrapper for the canonical ADMET drug filter service."""

from chem_inf_widgets.chemcore.admet.drug_filter_service import (
    DrugRow,
    FilterConfig,
    canonical_smiles,
    compute_drug_score,
    criteria_pass,
    filter_smiles,
    lipinski_stats,
    pains_match_info,
    selection_keep,
    veber_stats,
)


def canonicalize_smiles(smiles: str) -> str:
    """Backward-compatible alias for older imports."""
    return canonical_smiles(smiles)


__all__ = [
    "DrugRow",
    "FilterConfig",
    "canonical_smiles",
    "canonicalize_smiles",
    "compute_drug_score",
    "criteria_pass",
    "filter_smiles",
    "lipinski_stats",
    "pains_match_info",
    "selection_keep",
    "veber_stats",
]
