from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import shap as _shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression


# ── Feature-type detection ─────────────────────────────────────────────────

def _is_morgan_fp_cols(feature_cols: list[str]) -> bool:
    """True if column names look like fingerprint bits (Bit_N, fp_N, mfp_N, or plain ints)."""
    if not feature_cols:
        return False
    match = sum(
        1 for c in feature_cols
        if str(c).lower().startswith(("bit_", "fp_", "mfp_", "ecfp", "fcfp"))
        or str(c).isdigit()
    )
    return match / len(feature_cols) > 0.8


def _is_rdkit_desc_cols(feature_cols: list[str]) -> bool:
    _KNOWN = {
        "molwt", "mollogp", "tpsa", "numhdonors", "numhacceptors",
        "numrotatablebonds", "ringcount", "fractioncsp3", "heavyatomcount",
        "numaromaticrings", "numaliphaticrings", "labuteasa",
    }
    match = sum(1 for c in feature_cols if str(c).lower() in _KNOWN)
    return match >= 3


# ── SHAP helpers ───────────────────────────────────────────────────────────

def _get_model_step(pipeline: Any):
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model", pipeline)
    return pipeline


def _preprocess(pipeline: Any, X: np.ndarray) -> np.ndarray:
    """Apply all pipeline steps except the final model."""
    if not hasattr(pipeline, "steps"):
        return X
    Xt = X.copy()
    for name, step in pipeline.steps[:-1]:
        Xt = step.transform(Xt)
    return Xt


def compute_shap_values(
    pipeline: Any,
    X_explain: np.ndarray,
    X_background: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Returns (shap_values array of shape (n_samples, n_features), explainer_type_str).
    Always computed on raw features (before pipeline preprocessing) for readability,
    using the preprocessed arrays internally.
    """
    if not _SHAP_OK:
        return np.zeros_like(X_explain, dtype=float), "none"

    model = _get_model_step(pipeline)
    Xb_pre = _preprocess(pipeline, X_background)
    Xe_pre = _preprocess(pipeline, X_explain)

    if isinstance(model, (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor)):
        explainer = _shap.TreeExplainer(model, data=Xb_pre, feature_perturbation="interventional")
        sv = np.array(explainer.shap_values(Xe_pre))
        etype = "tree"
    elif isinstance(model, (Ridge, LinearRegression, ElasticNet, PLSRegression)):
        explainer = _shap.LinearExplainer(model, Xb_pre)
        sv = np.array(explainer.shap_values(Xe_pre))
        etype = "linear"
    else:
        # KernelExplainer – use tiny background to keep it fast
        bg = Xb_pre[:min(30, len(Xb_pre))]
        explainer = _shap.KernelExplainer(lambda x: pipeline.predict(x), bg)
        sv = np.array(explainer.shap_values(Xe_pre, nsamples=100))
        etype = "kernel"

    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return sv.astype(float), etype


# ── Per-atom contributions ─────────────────────────────────────────────────

def _atom_contribs_morgan(
    mol,
    shap_vec: np.ndarray,
    feature_cols: list[str],
    radius: int = 2,
) -> np.ndarray:
    """Map per-bit SHAP values back to atoms via Morgan bit info."""
    n_bits = len(feature_cols)
    bi: dict = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bi)
    n_atoms = mol.GetNumAtoms()
    contribs = np.zeros(n_atoms)
    for bit_idx, atom_list in bi.items():
        if bit_idx < len(shap_vec):
            sv = shap_vec[bit_idx]
            # Weight by number of atoms in each environment
            for atom_idx, _rad in atom_list:
                contribs[atom_idx] += sv / len(atom_list)
    return contribs


def _atom_contribs_rdkit_desc(
    mol,
    shap_vec: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    """
    For each RDKit descriptor with a known per-atom decomposition, distribute
    its SHAP weight proportionally to atomic contributions.
    """
    n_atoms = mol.GetNumAtoms()
    contribs = np.zeros(n_atoms)
    col_lower = [c.lower() for c in feature_cols]

    # LogP per-atom contributions
    if "mollogp" in col_lower:
        idx = col_lower.index("mollogp")
        sv = shap_vec[idx]
        try:
            lp_contribs = [x[0] for x in Crippen._GetAtomContribs(mol)]
            total = sum(abs(x) for x in lp_contribs) or 1.0
            for i, v in enumerate(lp_contribs):
                contribs[i] += sv * (v / total) if total else 0.0
        except Exception:
            pass

    # TPSA per-atom contributions
    if "tpsa" in col_lower:
        idx = col_lower.index("tpsa")
        sv = shap_vec[idx]
        try:
            tp_contribs = list(rdMolDescriptors._CalcTPSAContribs(mol))
            total = sum(abs(x) for x in tp_contribs) or 1.0
            for i, v in enumerate(tp_contribs):
                contribs[i] += sv * (v / total) if total else 0.0
        except Exception:
            pass

    # Labute ASA per-atom contributions
    if "labuteasa" in col_lower:
        idx = col_lower.index("labuteasa")
        sv = shap_vec[idx]
        try:
            asa_raw = rdMolDescriptors._CalcLabuteASAContribs(mol)
            asa_contribs = list(asa_raw[0])
            total = sum(abs(x) for x in asa_contribs) or 1.0
            for i, v in enumerate(asa_contribs):
                contribs[i] += sv * (v / total) if total else 0.0
        except Exception:
            pass

    # HeavyAtomCount: each non-H atom contributes equally
    if "heavyatomcount" in col_lower:
        idx = col_lower.index("heavyatomcount")
        sv = shap_vec[idx]
        for i in range(n_atoms):
            if mol.GetAtomWithIdx(i).GetAtomicNum() != 1:
                contribs[i] += sv / max(1, n_atoms)

    # NumHDonors / NumHAcceptors: distribute to N/O atoms
    for col_name, atomic_nums in [("numhdonors", {7, 8}), ("numhacceptors", {7, 8})]:
        if col_name in col_lower:
            idx = col_lower.index(col_name)
            sv = shap_vec[idx]
            targets = [i for i in range(n_atoms) if mol.GetAtomWithIdx(i).GetAtomicNum() in atomic_nums]
            if targets:
                for i in targets:
                    contribs[i] += sv / len(targets)

    # RingCount / NumAromaticRings: distribute to ring atoms
    for col_name in ("ringcount", "numaromaticrings", "numaliphaticrings"):
        if col_name in col_lower:
            idx = col_lower.index(col_name)
            sv = shap_vec[idx]
            ring_atoms = [i for i in range(n_atoms) if mol.GetAtomWithIdx(i).IsInRing()]
            if ring_atoms:
                for i in ring_atoms:
                    contribs[i] += sv / len(ring_atoms)

    return contribs


def compute_atom_contributions(
    smiles: str,
    shap_vec: np.ndarray,
    feature_cols: list[str],
    fp_radius: int = 2,
) -> Optional[np.ndarray]:
    """
    Returns per-atom contribution array (n_atoms,) or None if molecule invalid.
    Automatically chooses Morgan-bit or RDKit-descriptor path.
    """
    if not _RDKIT_OK:
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    if _is_morgan_fp_cols(feature_cols):
        return _atom_contribs_morgan(mol, shap_vec, feature_cols, fp_radius)
    if _is_rdkit_desc_cols(feature_cols):
        return _atom_contribs_rdkit_desc(mol, shap_vec, feature_cols)
    # Generic fallback: uniform SHAP magnitude distributed to all atoms
    n_atoms = mol.GetNumAtoms()
    score = float(np.sum(np.abs(shap_vec)))
    return np.full(n_atoms, score / max(1, n_atoms))


# ── Rendering ─────────────────────────────────────────────────────────────

def render_atom_heatmap(
    smiles: str,
    atom_contribs: np.ndarray,
    width: int = 400,
    height: int = 280,
) -> Optional[bytes]:
    """
    Render molecule with atoms colored by contribution.
    Green = contributes to higher prediction, red = lower.
    Returns SVG bytes or None.
    """
    if not _RDKIT_OK:
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None

    n = mol.GetNumAtoms()
    contribs = np.array(atom_contribs[:n], dtype=float)
    max_abs = float(np.max(np.abs(contribs))) or 1.0
    norm = contribs / max_abs  # in [-1, +1]

    atom_colors: dict = {}
    highlight_atoms: list[int] = []
    for i, v in enumerate(norm):
        highlight_atoms.append(i)
        t = abs(v)
        if v >= 0:
            # Green: (low-r, high-g, low-b) — intensity by t
            atom_colors[i] = (0.9 - 0.7 * t, 0.95, 0.9 - 0.7 * t)
        else:
            # Red
            atom_colors[i] = (0.95, 0.9 - 0.7 * t, 0.9 - 0.7 * t)

    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=[],
            highlightBondColors={},
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText().encode()
    except Exception:
        return None


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class AtomContributionResult:
    smiles: str
    compound_id: str
    prediction: float
    shap_values: np.ndarray          # shape (n_features,)
    feature_names: list[str]
    atom_contributions: Optional[np.ndarray]  # shape (n_atoms,) or None
    svg_bytes: Optional[bytes]
    explainer_type: str
    baseline: float                  # SHAP expected value


def explain_molecule(
    smiles: str,
    compound_id: str,
    pipeline: Any,
    X_row: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str],
    fp_radius: int = 2,
) -> AtomContributionResult:
    """Compute SHAP + atom contributions for a single molecule."""
    prediction = float(pipeline.predict(X_row.reshape(1, -1)).ravel()[0])

    if _SHAP_OK and len(X_background) >= 2:
        sv_all, etype = compute_shap_values(pipeline, X_row.reshape(1, -1), X_background)
        sv = sv_all[0]
        # baseline = prediction - sum(shap_values) ~ expected value
        baseline = prediction - float(np.sum(sv))
    else:
        sv = np.zeros(len(feature_names))
        etype = "none"
        baseline = float(np.nanmean(pipeline.predict(X_background)))

    atom_c = compute_atom_contributions(smiles, sv, feature_names, fp_radius)
    svg = render_atom_heatmap(smiles, atom_c) if atom_c is not None else None

    return AtomContributionResult(
        smiles=smiles,
        compound_id=compound_id,
        prediction=prediction,
        shap_values=sv,
        feature_names=feature_names,
        atom_contributions=atom_c,
        svg_bytes=svg,
        explainer_type=etype,
        baseline=baseline,
    )
