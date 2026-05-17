from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles, safe_mol_from_smiles


def canonical_smiles(smiles: str) -> str:
    """
    Canonical SMILES without explicit hydrogens.
    """
    if not smiles:
        return ""

    parsed = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True)
    if parsed.mol is None:
        return ""

    return safe_canonical_smiles(
        parsed.mol,
        remove_hs=False,
        canonical=True,
        isomeric=True,
    )
