from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.QED import qed

from chem_inf_widgets.chemcore.services.rdkit_safe import (
    safe_canonical_smiles,
    safe_mol_from_smiles,
)


logger = logging.getLogger(__name__)

SelectionMode = str  # "Forward All Molecules" | "Within Criteria" | "Out of Criteria"
FilterRule = str     # "Lipinski" | "Veber" | "Lipinski + Veber" | "None"
_PAINS_RESOURCE_PACKAGES = (
    "chem_inf_widgets.chemcore.data",
)


@dataclass(frozen=True)
class FilterConfig:
    filter_rule: FilterRule = "Lipinski + Veber"
    selection_mode: SelectionMode = "Within Criteria"
    compute_qed: bool = True
    compute_pains: bool = True
    highlight_pains_atoms: bool = False

    # Lipinski thresholds
    max_mw: float = 500.0
    max_logp: float = 5.0
    max_hbd: int = 5
    max_hba: int = 10

    # Veber thresholds
    max_rotb: int = 10
    max_tpsa: float = 140.0

    # Used only if resource auto-load fails
    pains_json_path: str = "smartspains.json"


@dataclass(frozen=True)
class DrugRow:
    smiles: str
    canonical_smiles: str

    mw: float
    logp: float
    hbd: float
    hba: float
    rotatable_bonds: float
    tpsa: float

    qed_score: float
    lipinski_violations: float
    veber_rule: float
    pains_match: float
    reactivity: float
    drug_score: float

    pains_regid: str
    criteria: str
    highlighted_atoms: str = ""


def canonical_smiles(smiles: str) -> str:
    """Canonical SMILES without explicit hydrogens (default RDKit behavior)."""
    s = (smiles or "").strip()
    if not s:
        return ""
    m = _mol_from_smiles(s)
    if m is None:
        return s
    return safe_canonical_smiles(m, remove_hs=False) or s


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not smiles:
        return None
    return safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol


# ---------------- PAINS loading + matching ----------------

@lru_cache(maxsize=1)
def _load_pains_rules_from_resources() -> List[Dict[str, str]]:
    """
    Load smartspains.json from the canonical package resource location.
    """
    try:
        import importlib.resources as ir
    except ImportError:
        logger.debug("importlib.resources is not available for PAINS resource loading.")
        return []

    for pkg in _PAINS_RESOURCE_PACKAGES:
        try:
            p = ir.files(pkg).joinpath("smartspains.json")
            if p.is_file():
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                logger.warning("PAINS resource in %s is not a JSON list.", pkg)
        except ModuleNotFoundError:
            logger.debug("PAINS resource package %s is not importable.", pkg)
        except FileNotFoundError:
            logger.debug("PAINS resource smartspains.json was not found in %s.", pkg)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode PAINS resource from %s: %s", pkg, exc)
        except OSError as exc:
            logger.warning("Failed to read PAINS resource from %s: %s", pkg, exc)
    return []


@lru_cache(maxsize=4)
def _load_pains_rules_from_path(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode PAINS rules from %s: %s", path, exc)
        return []
    except OSError as exc:
        logger.warning("Failed to read PAINS rules from %s: %s", path, exc)
        return []


@lru_cache(maxsize=8192)
def _mol_from_smarts(smarts: str) -> Optional[Chem.Mol]:
    if not smarts:
        return None
    try:
        with rdBase.BlockLogs():
            return Chem.MolFromSmarts(smarts)
    except (RuntimeError, ValueError) as exc:
        logger.debug("Invalid SMARTS pattern %r skipped: %s", smarts, exc)
        return None


@lru_cache(maxsize=1)
def _compiled_pains_rules_default() -> List[Tuple[Chem.Mol, str]]:
    rules = _load_pains_rules_from_resources()
    compiled: List[Tuple[Chem.Mol, str]] = []
    for r in rules:
        smarts = r.get("SMARTS") or ""
        regid = r.get("regID") or "PAINS"
        patt = _mol_from_smarts(smarts)
        if patt is not None:
            compiled.append((patt, regid))
    return compiled


def _compiled_pains_rules(cfg: FilterConfig) -> List[Tuple[Chem.Mol, str]]:
    compiled = _compiled_pains_rules_default()
    if compiled:
        return compiled

    # fallback to user-provided path
    rules = _load_pains_rules_from_path(cfg.pains_json_path)
    compiled2: List[Tuple[Chem.Mol, str]] = []
    for r in rules:
        smarts = r.get("SMARTS") or ""
        regid = r.get("regID") or "PAINS"
        patt = _mol_from_smarts(smarts)
        if patt is not None:
            compiled2.append((patt, regid))
    return compiled2


def pains_match_info(mol: Chem.Mol, cfg: FilterConfig) -> Tuple[float, str, str]:
    """
    Returns (pains_flag, regid_csv, atom_indices_csv).
    If highlight_pains_atoms=False -> atom_indices_csv = "".
    """
    compiled = _compiled_pains_rules(cfg)
    if not compiled:
        return 0.0, "None", ""

    regids: List[str] = []
    atoms = set()

    for patt, regid in compiled:
        if not mol.HasSubstructMatch(patt):
            continue
        regids.append(regid)
        if cfg.highlight_pains_atoms:
            for match in mol.GetSubstructMatches(patt):
                atoms.update(match)

    if not regids:
        return 0.0, "None", ""

    regid_csv = ", ".join(sorted(set(regids)))
    atom_csv = ", ".join(str(i) for i in sorted(atoms)) if (cfg.highlight_pains_atoms and atoms) else ""
    return 1.0, regid_csv, atom_csv


# ---------------- Rules / scoring ----------------

def lipinski_stats(mol: Chem.Mol, cfg: FilterConfig) -> Tuple[int, float, float, float, float]:
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd = float(rdMolDescriptors.CalcNumHBD(mol))
    hba = float(rdMolDescriptors.CalcNumHBA(mol))

    vio = 0
    if mw > cfg.max_mw:
        vio += 1
    if logp > cfg.max_logp:
        vio += 1
    if hbd > cfg.max_hbd:
        vio += 1
    if hba > cfg.max_hba:
        vio += 1
    return vio, mw, logp, hbd, hba


def veber_stats(mol: Chem.Mol, cfg: FilterConfig) -> Tuple[bool, float, float]:
    rotb = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    ok = (rotb <= cfg.max_rotb) and (tpsa <= cfg.max_tpsa)
    return ok, rotb, tpsa


def criteria_pass(cfg: FilterConfig, lip_vio: int, veber_ok: bool) -> bool:
    fr = (cfg.filter_rule or "None").strip()
    if fr == "Lipinski":
        return lip_vio <= 1
    if fr == "Veber":
        return bool(veber_ok)
    if fr == "Lipinski + Veber":
        return (lip_vio <= 1) and bool(veber_ok)
    return True


def selection_keep(cfg: FilterConfig, passed: bool) -> bool:
    sm = (cfg.selection_mode or "Forward All Molecules").strip()
    if sm == "Within Criteria":
        return passed
    if sm == "Out of Criteria":
        return not passed
    return True


def compute_drug_score(qed_score: float, lip_vio: int, pains: bool, veber_ok: bool, reactivity: bool) -> float:
    """
    Simple composite score:
      start from QED, subtract penalties.
    """
    score = float(qed_score)
    if lip_vio > 1:
        score -= 0.2
    if pains:
        score -= 0.3
    if not veber_ok:
        score -= 0.1
    if reactivity:
        score -= 0.2
    return max(score, 0.0)


# ---------------- Main API ----------------

def filter_smiles(
    smiles_list: Sequence[str],
    cfg: FilterConfig,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[DrugRow]:
    """
    Compute descriptors + apply filters.
    Returns forwarded rows (based on cfg.selection_mode).
    """
    out: List[DrugRow] = []
    n = len(smiles_list)

    for i, smi in enumerate(smiles_list):
        if progress_cb:
            progress_cb(i + 1, n)

        smi0 = (str(smi) if smi is not None else "").strip()
        if not smi0:
            continue

        mol = _mol_from_smiles(smi0)
        if mol is None:
            continue

        can = canonical_smiles(smi0)

        lip_vio, mw, logp, hbd, hba = lipinski_stats(mol, cfg)
        veber_ok, rotb, tpsa = veber_stats(mol, cfg)

        qed_score = float(qed(mol)) if cfg.compute_qed else float("nan")

        pains_flag, pains_regid, pains_atoms = (0.0, "None", "")
        if cfg.compute_pains:
            pains_flag, pains_regid, pains_atoms = pains_match_info(mol, cfg)

        reactive = False  # placeholder for later
        passed = criteria_pass(cfg, lip_vio, veber_ok)

        if not selection_keep(cfg, passed):
            continue

        drug_score = compute_drug_score(
            qed_score=qed_score if not math.isnan(qed_score) else 0.0,
            lip_vio=lip_vio,
            pains=bool(pains_flag),
            veber_ok=veber_ok,
            reactivity=reactive,
        )

        out.append(
            DrugRow(
                smiles=smi0,
                canonical_smiles=can,
                mw=mw,
                logp=logp,
                hbd=hbd,
                hba=hba,
                rotatable_bonds=rotb,
                tpsa=tpsa,
                qed_score=qed_score,
                lipinski_violations=float(lip_vio),
                veber_rule=1.0 if veber_ok else 0.0,
                pains_match=float(pains_flag),
                reactivity=0.0,
                drug_score=float(drug_score),
                pains_regid=pains_regid,
                criteria="Pass" if passed else "Fail",
                highlighted_atoms=pains_atoms,
            )
        )

    return out
