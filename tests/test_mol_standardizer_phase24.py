from __future__ import annotations

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import (
    DROPPED_REASON,
    QC_FLAGS,
    STANDARDIZATION_CHANGED,
    STANDARDIZATION_INPUT_SMILES,
    STANDARDIZATION_LOG,
    STANDARDIZATION_OUTPUT_SMILES,
    STANDARDIZATION_PROFILE,
    STANDARDIZATION_STATUS,
    STANDARDIZATION_VERSION_FIELD,
    STANDARDIZED_SMILES,
    TRANSFORM_LOG,
)
from chem_inf_widgets.chemcore.services.mol_standardizer import (
    STANDARDIZATION_PRESETS,
    MolStandardizer,
    get_standardization_config,
)


def test_phase24_user_facing_profiles_exist():
    for profile in ["minimal", "qsar_ready", "chembl_like", "docking_ready"]:
        assert profile in STANDARDIZATION_PRESETS
        assert get_standardization_config(profile) is STANDARDIZATION_PRESETS[profile]


def test_qsar_ready_profile_removes_salt_and_writes_audit_fields():
    cm = ChemMol.from_smiles("CCO.Cl", name="ethanol_salt")
    cm.set_prop("SMILES", "CCO.Cl")

    out_mols, results = MolStandardizer(profile="qsar_ready").standardize_chemmols([cm])

    assert results[0].ok
    out = out_mols[0]
    assert out.get_prop(STANDARDIZED_SMILES) == "CCO"
    assert out.get_prop(STANDARDIZATION_STATUS) == "ok"
    assert out.get_prop(STANDARDIZATION_PROFILE) == "qsar_ready"
    assert out.get_prop(STANDARDIZATION_INPUT_SMILES) == "CCO.Cl"
    assert out.get_prop(STANDARDIZATION_OUTPUT_SMILES) == "CCO"
    assert out.get_prop(STANDARDIZATION_CHANGED) is True
    assert out.get_prop(STANDARDIZATION_VERSION_FIELD) == "phase2.4"
    assert out.get_prop(STANDARDIZATION_LOG)
    assert "standardize_qsar_ready" in out.get_prop(TRANSFORM_LOG)


def test_failed_standardization_marks_shared_contract_fields():
    cm = ChemMol.from_smiles("CCO", name="ethanol")
    cm.set_prop("SMILES", "not_a_smiles")

    out_mols, results = MolStandardizer(profile="qsar_ready").standardize_chemmols([cm])

    assert not results[0].ok
    assert out_mols[0].get_prop(STANDARDIZATION_STATUS) == "failed"
    assert "standardization_failed" in out_mols[0].get_prop(QC_FLAGS)
    assert out_mols[0].get_prop(DROPPED_REASON) == "standardization_failed"


def test_docking_ready_preserves_formal_charge_better_than_qsar_ready():
    charged = "C[NH+](C)C"

    qsar = MolStandardizer(profile="qsar_ready").standardize_smiles(charged)
    docking = MolStandardizer(profile="docking_ready").standardize_smiles(charged)

    assert qsar.ok
    assert docking.ok
    assert "+" not in qsar.output_smiles
    assert "+" in docking.output_smiles
