from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from rdkit import Chem

from chem_inf_widgets.chemcore.mol import ChemMol


@dataclass
class SdfReadResult:
    mols: List[ChemMol]
    n_total: int
    n_failed: int


def _extract_props(mol: Chem.Mol, keep_props: Optional[Sequence[str]] = None) -> Dict[str, str]:
    names = list(mol.GetPropNames(includePrivate=False, includeComputed=False))
    if keep_props is not None:
        allowed = set(keep_props)
        names = [n for n in names if n in allowed]

    props: Dict[str, str] = {}
    for n in names:
        try:
            props[n] = mol.GetProp(n)
        except Exception:
            continue
    return props


def read_sdf(
    path: Union[str, Path],
    *,
    sanitize: bool = True,
    remove_hs: bool = True,
    name_prop: str = "_Name",
    keep_props: Optional[Sequence[str]] = None,
    max_mols: int | None = None,
) -> SdfReadResult:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    supplier = Chem.SDMolSupplier(str(p), sanitize=sanitize, removeHs=False)

    mols: List[ChemMol] = []
    n_total = 0
    n_failed = 0

    for mol in supplier:
        n_total += 1
        if mol is None:
            n_failed += 1
            continue

        try:
            if remove_hs:
                mol = Chem.RemoveHs(mol, sanitize=sanitize)

            name: Optional[str] = None
            if name_prop and mol.HasProp(name_prop):
                v = mol.GetProp(name_prop).strip()
                name = v or None

            props = _extract_props(mol, keep_props=keep_props)

            mols.append(ChemMol(mol=mol, name=name, props=props, cache={}))

        except Exception:
            n_failed += 1
            continue

        if max_mols is not None and len(mols) >= max_mols:
            break

    return SdfReadResult(mols=mols, n_total=n_total, n_failed=n_failed)


def write_sdf(
    mols: Iterable[ChemMol],
    path: Union[str, Path],
    *,
    include_props: bool | Iterable[str] = True,
    write_name: bool = True,
) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(p))
    n_written = 0

    if include_props is True:
        allowed = None
    elif include_props is False:
        allowed = set()
    else:
        allowed = set(include_props)

    for cm in mols:
        mol = cm.mol
        if mol is None:
            continue

        if write_name and cm.name:
            try:
                mol.SetProp("_Name", str(cm.name))
            except Exception:
                pass

        if allowed is None:
            items = cm.props.items()
        else:
            items = ((k, v) for (k, v) in cm.props.items() if k in allowed)

        for k, v in items:
            try:
                mol.SetProp(str(k), "" if v is None else str(v))
            except Exception:
                continue

        writer.write(mol)
        n_written += 1

    writer.close()
    return n_written

