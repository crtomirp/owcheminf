from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, MACCSkeys, rdmolops

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles

try:
    # New-ish API (preferred)
    from rdkit.Chem import rdFingerprintGenerator as _rd_fpg
except Exception:  # pragma: no cover
    _rd_fpg = None  # type: ignore

try:
    from rdkit.Avalon import pyAvalonTools
except Exception:  # pragma: no cover
    pyAvalonTools = None  # type: ignore


FPType = str  # "morgan" | "rdkit" | "maccs" | "avalon"


@dataclass(frozen=True)
class FingerprintResult:
    X: np.ndarray  # shape (n_valid, n_bits)
    smiles: List[str]
    valid_indices: List[int]
    failed_indices: List[int]
    bit_names: List[str]
    fp_type: FPType
    bit_size: int = 0
    radius: int = 0
    params: Optional[Dict[str, object]] = None
    errors: Optional[List[str]] = None


def _mol_from_smiles(
    smiles: str,
    *,
    sanitize: bool = True,
    remove_hs: bool = False,
) -> tuple[Optional[Chem.Mol], str]:
    parsed = safe_mol_from_smiles(
        smiles,
        sanitize=sanitize,
        remove_hs=remove_hs,
    )
    if parsed.mol is None:
        return None, parsed.error or "Invalid molecule."
    if parsed.warnings:
        return parsed.mol, "; ".join(parsed.warnings)
    return parsed.mol, ""


def _bitvect_to_numpy(fp) -> np.ndarray:
    nbits = int(fp.GetNumBits())
    arr = np.zeros((nbits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


@lru_cache(maxsize=32)
def _get_morgan_gen(radius: int, bit_size: int):
    # Preferred generator (no deprecation warnings)
    if _rd_fpg is not None:
        try:
            return _rd_fpg.GetMorganGenerator(radius=int(radius), fpSize=int(bit_size))
        except AttributeError:
            pass
    return None


@lru_cache(maxsize=32)
def _get_rdkit_gen(bit_size: int):
    if _rd_fpg is not None:
        try:
            return _rd_fpg.GetRDKitFPGenerator(fpSize=int(bit_size))
        except AttributeError:
            pass
    return None


def _compute_fp(
    mol: Chem.Mol,
    fp_type: FPType,
    *,
    bit_size: int,
    radius: int,
) -> Optional[np.ndarray]:
    if mol is None:
        return None

    t = fp_type.lower().strip()

    if t == "morgan":
        # New API first
        gen = _get_morgan_gen(radius=radius, bit_size=bit_size)
        if gen is not None:
            try:
                fp = gen.GetFingerprint(mol)
                return _bitvect_to_numpy(fp)
            except Exception:
                pass

        # Fallback (older RDKit) - Suppress Deprecation Warnings
        blocker = rdBase.BlockLogs()
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(bit_size))
            return _bitvect_to_numpy(fp)
        finally:
            del blocker

    if t == "rdkit":
        gen = _get_rdkit_gen(bit_size=bit_size)
        if gen is not None:
            try:
                fp = gen.GetFingerprint(mol)
                return _bitvect_to_numpy(fp)
            except Exception:
                pass

        fp = rdmolops.RDKFingerprint(mol, fpSize=int(bit_size))
        return _bitvect_to_numpy(fp)

    if t == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)  # 167 bits
        return _bitvect_to_numpy(fp)

    if t == "avalon":
        if pyAvalonTools is None:
            return None
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=int(bit_size))
        return _bitvect_to_numpy(fp)

    raise ValueError(f"Unsupported fingerprint type: {fp_type!r}")


def compute_fingerprints_from_smiles(
    smiles_list: Sequence[str],
    *,
    fp_type: FPType = "morgan",
    bit_size: int = 1024,
    radius: int = 2,
    sanitize: bool = True,
    remove_hs: bool = False,
    remove_low_variance: bool = False,
    variance_threshold: float = 0.01,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> FingerprintResult:
    bit_size = int(bit_size)
    radius = int(radius)

    valid_idx: List[int] = []
    failed_idx: List[int] = []
    errors: List[str] = []
    rows: List[np.ndarray] = []
    valid_smiles: List[str] = []

    n = len(smiles_list)
    for i, smi in enumerate(smiles_list):
        if cancel_cb is not None and cancel_cb():
            break

        mol, parse_msg = _mol_from_smiles(str(smi), sanitize=sanitize, remove_hs=remove_hs)

        try:
            fp_arr = _compute_fp(mol, fp_type, bit_size=bit_size, radius=radius) if mol else None
        except Exception as exc:
            fp_arr = None
            parse_msg = f"Fingerprint computation failed: {exc}"

        if fp_arr is None:
            failed_idx.append(i)
            errors.append(parse_msg or "Fingerprint computation failed.")
        else:
            valid_idx.append(i)
            rows.append(fp_arr)
            valid_smiles.append(str(smi))

        if progress_cb is not None:
            progress_cb(int((i + 1) * 100 / max(1, n)))

    if not rows:
        return FingerprintResult(
            X=np.zeros((0, 0), dtype=np.float32),
            smiles=[],
            valid_indices=[],
            failed_indices=list(range(len(smiles_list))),
            bit_names=[],
            fp_type=fp_type,
            bit_size=0,
            radius=radius,
            params={
                "fp_type": fp_type,
                "requested_bit_size": bit_size,
                "effective_bit_size": 0,
                "radius": radius,
                "sanitize": sanitize,
                "remove_hs": remove_hs,
                "remove_low_variance": remove_low_variance,
                "variance_threshold": variance_threshold,
            },
            errors=errors or ["No valid molecules."],
        )

    X = np.vstack(rows).astype(np.float32)
    fp_name = str(fp_type).lower().strip()
    if fp_name == "maccs":
        bit_names = [f"MACCS_{j:03d}" for j in range(X.shape[1])]
    else:
        bit_names = [f"{fp_name}_{j:04d}" for j in range(X.shape[1])]

    if remove_low_variance and X.shape[1] > 0:
        variances = np.var(X, axis=0)
        keep = variances >= float(variance_threshold)
        if np.any(keep):
            X = X[:, keep]
            bit_names = [name for name, k in zip(bit_names, keep) if k]
        else:
            X = X[:, :1]
            bit_names = bit_names[:1]

    return FingerprintResult(
        X=X,
        smiles=valid_smiles,
        valid_indices=valid_idx,
        failed_indices=failed_idx,
        bit_names=bit_names,
        fp_type=fp_type,
        bit_size=int(X.shape[1]),
        radius=radius,
        params={
            "fp_type": fp_type,
            "requested_bit_size": bit_size,
            "effective_bit_size": int(X.shape[1]),
            "radius": radius,
            "sanitize": sanitize,
            "remove_hs": remove_hs,
            "remove_low_variance": remove_low_variance,
            "variance_threshold": variance_threshold,
        },
        errors=errors,
    )
