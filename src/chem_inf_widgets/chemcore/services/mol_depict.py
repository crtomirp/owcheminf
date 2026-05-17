from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Draw

from Orange.data import Table

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


# =========================
# Data container
# =========================

@dataclass
class DepictItem:
    mol: Chem.Mol
    title: str
    props: dict
    highlight_atoms: list[int]


# =========================
# Helpers
# =========================

def _find_smiles_meta(table: Table):
    for v in table.domain.metas:
        if (v.name or "").strip().lower() == "smiles":
            return v
    return None


def _parse_highlight_atoms(props: dict) -> list[int]:
    """
    Parse 'Highlighted Atoms' meta (CSV: '0,1,5') into a list[int].
    """
    raw = props.get("Highlighted Atoms") or props.get("highlighted atoms") or ""
    s = str(raw).strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    # unique + stable order
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


# =========================
# Table → items
# =========================

def table_to_items(
    table: Table,
    selected_props: Optional[list[str]] = None,
    highlight_enabled: bool = True,
) -> List[DepictItem]:
    smiles_var = _find_smiles_meta(table)
    if smiles_var is None:
        raise ValueError("SMILES meta column not found")

    smiles_col = table.get_column(smiles_var)

    items: list[DepictItem] = []
    for i, smi in enumerate(smiles_col):
        if not smi:
            continue

        mol = safe_mol_from_smiles(str(smi), sanitize=True, remove_hs=True).mol
        if mol is None:
            continue

        props = {}
        for v in table.domain.metas:
            val = table[i, v]
            if val is not None:
                props[v.name] = str(val)
        if selected_props:
            props = {
                k: v
                for k, v in props.items()
                if k in selected_props
                or k in {"Name", "SMILES", "ime spojine", "Highlighted Atoms"}
            }

        title = (
            props.get("Name")
            or props.get("ime spojine")
            or props.get("SMILES")
            or f"#{i+1}"
        )

        hl = _parse_highlight_atoms(props) if highlight_enabled else []

        items.append(
            DepictItem(
                mol=mol,
                title=title,
                props=props,
                highlight_atoms=hl,
            )
        )

    return items


# =========================
# ChemMol list → items
# =========================

def chemmols_to_items(
    mols: List[ChemMol],
    selected_props: Optional[list[str]] = None,
    highlight_enabled: bool = True,
) -> List[DepictItem]:
    items: list[DepictItem] = []
    for i, cm in enumerate(mols):
        if cm is None:
            continue

        mol = cm.mol
        if mol is None:
            smiles = str(cm.get_prop("SMILES") or "").strip()
            mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol if smiles else None
        if mol is None:
            continue

        title = cm.name or f"#{i+1}"

        props = dict(cm.props or {})
        if selected_props:
            props = {
                k: v
                for k, v in props.items()
                if k in selected_props or k == "Highlighted Atoms"
            }
        # allow highlight via props too (optional)
        hl = _parse_highlight_atoms(props) if highlight_enabled else []

        items.append(
            DepictItem(
                mol=mol,
                title=title,
                props=props,
                highlight_atoms=hl,
            )
        )

    return items


# =========================
# Rendering
# =========================

def render_mol_png(
    mol: Chem.Mol,
    size: int = 240,
    highlight_atoms: Optional[list[int]] = None,
    remove_hs_for_drawing: bool = True,
    use_rdcoordgen: bool = True,
) -> bytes:
    """
    Render molecule to PNG bytes.

    highlight_atoms:
        list of atom indices to highlight (substructure match).
    """
    if mol is None:
        return b""

    m = mol
    if remove_hs_for_drawing:
        try:
            m = Chem.RemoveHs(m)
        except Exception:
            m = mol

    try:
        if use_rdcoordgen:
            Chem.rdCoordGen.AddCoords(m)
        else:
            Chem.Compute2DCoords(m)
    except Exception:
        # if coordinate generation fails, still try draw
        pass

    drawer = Draw.MolDraw2DCairo(size, size)
    drawer.DrawMolecule(
        m,
        highlightAtoms=list(highlight_atoms or []),
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# =========================
# Pagination helper (kept)
# =========================

def page_slice(n: int, page: int, page_size: int):
    if n == 0:
        return 0, 0, 0, 0

    pages = max(1, (n + page_size - 1) // page_size)
    page = max(0, min(page, pages - 1))

    start = page * page_size
    end = min(start + page_size, n)

    return page, start, end, pages
