from __future__ import annotations

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import DROPPED_REASON, TRANSFORM_LOG
from chem_inf_widgets.chemcore.services.curation_summary import (
    CURATION_READY_FOR_QSAR,
    CURATION_STATUS,
    CURATION_VERSION_FIELD,
    QSAR_COMPATIBLE_STANDARDIZATION_PROFILES,
    annotate_curation_props,
    curation_summary_to_table,
    summary_from_standardization_rows,
)
from chem_inf_widgets.chemcore.services.mol_standardizer import MolStandardizer


def test_phase25_curation_summary_marks_qsar_ready_standardization():
    rows = [
        {"row_index": 1, "source": "molecules", "ok": 1, "profile": "qsar_ready"},
        {"row_index": 2, "source": "molecules", "ok": 0, "profile": "qsar_ready"},
    ]
    summary = summary_from_standardization_rows(rows, "qsar_ready")
    table = curation_summary_to_table(summary)

    assert "qsar_ready" in QSAR_COMPATIBLE_STANDARDIZATION_PROFILES
    assert summary.qsar_ready_records == 1
    assert summary.failed_records == 1
    # In Orange environments this is an Orange Table; in minimal CI without
    # Orange the helper safely returns None.
    if table is not None:
        assert len(table) > 0


def test_phase25_standardized_molecules_get_curation_contract_fields():
    mol = ChemMol.from_smiles("CCO.Cl", name="ethanol_salt")
    mol.set_prop("SMILES", "CCO.Cl")

    out_mols, results = MolStandardizer(profile="qsar_ready").standardize_chemmols([mol])

    # The core standardizer still keeps the chemistry audit; the widget adds
    # workflow curation fields.  Here we assert the new constants are available
    # for downstream widgets and reports.
    assert results[0].ok
    for key in (CURATION_STATUS, CURATION_READY_FOR_QSAR, CURATION_VERSION_FIELD):
        assert isinstance(key, str)
    assert out_mols[0].get_prop("standardization_status") == "ok"


def test_phase25_curation_annotation_adds_transform_step_and_drop_reason():
    mol = ChemMol.from_smiles("CCO", name="ethanol")

    annotated = annotate_curation_props(
        [mol],
        stage="qc",
        status="blocked",
        blockers="Invalid structure",
        recommended_next_step="Fix input",
    )

    assert annotated[0].get_prop(DROPPED_REASON) == "Invalid structure"
    assert "curation_qc" in annotated[0].get_prop(TRANSFORM_LOG)
