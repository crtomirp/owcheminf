from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from rdkit import Chem, rdBase


@dataclass(frozen=True)
class MolParseResult:
    ok: bool
    mol: Optional[Chem.Mol]
    input_smiles: str
    canonical_smiles: Optional[str]
    error: Optional[str]
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MolStandardizeResult:
    ok: bool
    mol: Optional[Chem.Mol]
    input_smiles: str
    standardized_smiles: Optional[str]
    error: Optional[str]
    warnings: List[str] = field(default_factory=list)
    log: str = ""


def validate_molecule(mol: Optional[Chem.Mol]) -> List[str]:
    warnings: List[str] = []
    if mol is None:
        return ["Molecule is None."]

    try:
        Chem.SanitizeMol(Chem.Mol(mol))
    except Exception as exc:
        warnings.append(f"Sanitize failed: {exc}")

    return warnings


def safe_canonical_smiles(
    mol: Optional[Chem.Mol],
    *,
    remove_hs: bool = True,
    canonical: bool = True,
    isomeric: bool = True,
) -> str:
    if mol is None:
        return ""

    try:
        mol_copy = Chem.Mol(mol)
        if remove_hs:
            try:
                mol_copy = Chem.RemoveHs(mol_copy)
            except Exception:
                pass
        return Chem.MolToSmiles(
            mol_copy,
            canonical=canonical,
            isomericSmiles=isomeric,
        )
    except Exception:
        return ""


def safe_mol_from_smiles(
    smiles: str,
    *,
    sanitize: bool = True,
    remove_hs: bool = True,
) -> MolParseResult:
    input_smiles = (smiles or "").strip()
    if not input_smiles:
        return MolParseResult(
            ok=False,
            mol=None,
            input_smiles="",
            canonical_smiles=None,
            error="Empty SMILES.",
            warnings=[],
        )

    warnings: List[str] = []
    try:
        blocker = rdBase.BlockLogs()
        try:
            mol = Chem.MolFromSmiles(input_smiles, sanitize=sanitize)
        finally:
            del blocker
    except Exception as exc:
        return MolParseResult(
            ok=False,
            mol=None,
            input_smiles=input_smiles,
            canonical_smiles=None,
            error=f"RDKit parse failed: {exc}",
            warnings=[],
        )

    if mol is None:
        return MolParseResult(
            ok=False,
            mol=None,
            input_smiles=input_smiles,
            canonical_smiles=None,
            error="Invalid SMILES.",
            warnings=[],
        )

    if remove_hs:
        try:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
        except TypeError:
            mol = Chem.RemoveHs(mol)
        except Exception as exc:
            warnings.append(f"RemoveHs failed: {exc}")

    warnings.extend(validate_molecule(mol))
    canonical_smiles = safe_canonical_smiles(mol, remove_hs=False)
    if not canonical_smiles:
        warnings.append("Could not derive canonical SMILES.")

    return MolParseResult(
        ok=True,
        mol=mol,
        input_smiles=input_smiles,
        canonical_smiles=canonical_smiles or None,
        error=None,
        warnings=warnings,
    )


def safe_mol_to_inchikey(mol: Optional[Chem.Mol]) -> str:
    if mol is None:
        return ""

    try:
        return Chem.MolToInchiKey(Chem.Mol(mol)) or ""
    except Exception:
        return ""


def safe_standardize_mol(
    mol: Optional[Chem.Mol],
    *,
    config=None,
) -> MolStandardizeResult:
    input_smiles = safe_canonical_smiles(mol)
    if mol is None:
        return MolStandardizeResult(
            ok=False,
            mol=None,
            input_smiles="",
            standardized_smiles=None,
            error="Molecule is None.",
            warnings=[],
            log="",
        )

    from chem_inf_widgets.chemcore.services.mol_standardizer import MolStandardizer

    service = MolStandardizer(config=config)
    result = service.standardize_smiles(input_smiles)
    warnings = validate_molecule(result.mol) if result.mol is not None else []

    return MolStandardizeResult(
        ok=result.ok,
        mol=result.mol,
        input_smiles=input_smiles,
        standardized_smiles=result.output_smiles or None,
        error=None if result.ok else result.log,
        warnings=warnings,
        log=result.log,
    )
