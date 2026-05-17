from __future__ import annotations

from typing import Iterable, List

from rdkit.Chem import Crippen, Descriptors

from chem_inf_widgets.chemcore.mol import ChemMol


def run_admet_filter(
    mols: Iterable[ChemMol],
    *,
    max_logp: float = 5.0,
    max_mw: float = 500.0,
) -> List[ChemMol]:
    """
    Simple ADMET filter.

    Stores results in ChemMol.props:
      - logP
      - MW
      - admet.pass
    """
    passed: List[ChemMol] = []

    for m in mols:
        mol = m.mol

        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)

        m.set_prop("logP", logp)
        m.set_prop("MW", mw)

        ok = (logp <= max_logp) and (mw <= max_mw)
        m.set_prop("admet.pass", ok)

        if ok:
            passed.append(m)

    return passed
