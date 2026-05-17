from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem

# Optional py3Dmol
try:
    import py3Dmol  # type: ignore

    _HAS_PY3DMOL = True
except Exception:
    py3Dmol = None  # type: ignore
    _HAS_PY3DMOL = False


@dataclass(frozen=True)
class Viewer3DConfig:
    width: int = 820
    height: int = 620
    style: str = "stick"  # stick|sphere|line
    surface: bool = False
    surface_opacity: float = 0.6
    add_hs: bool = True
    optimize: bool = True
    max_opt_iters: int = 200


def has_3d_conformer(mol: Chem.Mol) -> bool:
    """True if the molecule has a 3D conformer (not just 2D coordinates)."""
    if mol is None:
        return False
    if mol.GetNumConformers() == 0:
        return False

    conf = mol.GetConformer()
    if conf.Is3D():
        return True

    # Some mols have Is3D False but still have z != 0
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        if abs(float(p.z)) > 1e-6:
            return True
    return False


def _mmff_ok(m: Chem.Mol) -> bool:
    try:
        props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
        return props is not None
    except Exception:
        return False


def _embed_3d(m: Chem.Mol) -> None:
    """Embed a single 3D conformer with robust fallbacks."""
    seed = 0xC0FFEE

    res: int = 1

    # Preferred: ETKDGv3
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        res = AllChem.EmbedMolecule(m, params)
        if res != 0:
            # Rescue for difficult cases
            params2 = AllChem.ETKDGv3()
            params2.randomSeed = seed
            params2.useRandomCoords = True
            res = AllChem.EmbedMolecule(m, params2)
    except Exception:
        res = 1

    # Fallback: ETKDG (older)
    if res != 0:
        try:
            params = AllChem.ETKDG()
            try:
                params.randomSeed = seed
            except Exception:
                pass
            res = AllChem.EmbedMolecule(m, params)
        except Exception:
            res = 1

    # Final fallback: useRandomCoords signature
    if res != 0:
        try:
            res = AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=seed)
        except TypeError:
            res = AllChem.EmbedMolecule(m, useRandomCoords=True)

    if res != 0:
        raise ValueError("RDKit failed to generate a 3D conformer (EmbedMolecule).")


def ensure_3d(mol: Chem.Mol, cfg: Viewer3DConfig) -> Chem.Mol:
    """Return a copy with a proper 3D conformer and (optionally) well-placed H.

    Fix for "weird hydrogen positions": many incoming molecules carry **2D**
    coordinates (they have a conformer, but it's not 3D). If you treat "has a
    conformer" as "already 3D" and then add hydrogens with coordinates, H atoms
    end up placed on a flat/2D geometry and look wrong in a 3D viewer.

    Rules:
      - If the molecule does NOT have a real 3D conformer, re-embed to 3D.
      - When generating 3D and cfg.add_hs=True, add H **before** embedding so
        ETKDG places everything (including H) consistently in 3D.
      - When a true 3D conformer already exists, add H with addCoords=True.
    """
    if mol is None:
        raise ValueError("No molecule")

    m = Chem.Mol(mol)

    needs_embedding = not has_3d_conformer(m)
    if needs_embedding:
        # Remove existing (typically 2D) conformers to avoid reusing flat coords
        if m.GetNumConformers() > 0:
            m.RemoveAllConformers()

        if cfg.add_hs:
            # Add H BEFORE embedding: ETKDG will place H in correct 3D positions
            m = Chem.AddHs(m)

        _embed_3d(m)

    else:
        # Already 3D
        if cfg.add_hs:
            # Add H while keeping the existing 3D geometry
            m = Chem.AddHs(m, addCoords=True)

    if cfg.optimize:
        try:
            if _mmff_ok(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=int(cfg.max_opt_iters))
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=int(cfg.max_opt_iters))
        except Exception:
            # Optimization is best-effort; embedding is the critical part.
            pass

    return m


def build_3d_html_from_mol(mol: Chem.Mol, cfg: Viewer3DConfig) -> str:
    """Build HTML for QWebEngineView using py3Dmol (Python-side)."""
    if not _HAS_PY3DMOL:
        raise ImportError("py3Dmol is not installed. Install: pip install py3Dmol")

    m3d = ensure_3d(mol, cfg)
    mb = Chem.MolToMolBlock(m3d)

    v = py3Dmol.view(width=int(cfg.width), height=int(cfg.height))
    v.addModel(mb, "mol")

    st = cfg.style.lower().strip()
    if st == "sphere":
        v.setStyle({"sphere": {}})
    elif st == "line":
        v.setStyle({"line": {}})
    else:
        v.setStyle({"stick": {}})

    if cfg.surface:
        try:
            v.addSurface("VDW", {"opacity": float(cfg.surface_opacity)})
        except Exception:
            pass

    v.zoomTo()
    return v._make_html()


def pick_smiles_from_table_row_str(row_values: Sequence[object]) -> Optional[str]:
    """Heuristic helper to pick a SMILES-like string from a row of values."""
    for v in row_values:
        if v is None:
            continue
        s = str(v).strip()
        if s and any(ch.isalpha() for ch in s) and (
            "/" in s or "=" in s or "(" in s or ")" in s or "C" in s
        ):
            return s
    return None
