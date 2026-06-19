"""Widget-level tests for the CmDock Docking node (no real CmDock needed)."""
import os
import tempfile
import unittest

import numpy as np
from Orange.data import Domain, StringVariable, Table
from Orange.widgets.tests.base import WidgetTest
from rdkit import Chem
from rdkit.Chem import AllChem

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.widgets.ow_cmdock_docking import (
    OWCmDockDocking,
    build_poses_table,
    molecule_to_sdf_record,
    poses_to_chemmols,
)


def _mol_3d(smiles="CCO", name="lig"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol


def _row(status="ok", score=-7.5, props=None, **extra):
    row = {k: None for k in (
        "input_row_id", "input_index", "ligand_name", "pose_index", "rank_by_score",
        "status", "score", "score_field", "pose_sdf", "result_sdf_path",
        "cmdock_command", "cmdock_return_code", "cmdock_stdout", "cmdock_stderr",
        "error_message", "properties_json",
    )}
    row.update(status=status, score=score, pose_index=1, rank_by_score=1,
               ligand_name="Lig", cmdock_return_code=0, _properties=props or {})
    row.update(extra)
    return row


class BuildPosesTableTest(WidgetTest):
    def test_numeric_props_become_attributes_text_become_metas(self):
        rows = [_row(props={"SCORE": "-9.1", "SCORE.INTER": "-7.5", "NAME": "aspirin"},
                     score_field="SCORE.INTER", compound_id="CMPD-1")]
        table = build_poses_table(rows, ["compound_id"])
        attrs = {v.name for v in table.domain.attributes}
        metas = {v.name for v in table.domain.metas}

        self.assertIn("SCORE", attrs)          # numeric SDF property -> attribute
        self.assertIn("SCORE.INTER", attrs)
        # the ranking column is named after the chosen field, not plain "score"
        self.assertIn("ranking_score (SCORE.INTER)", attrs)
        self.assertNotIn("score", attrs)
        self.assertIn("NAME", metas)           # text SDF property -> meta
        self.assertIn("pose_sdf", metas)
        self.assertIn("compound_id", metas)    # pass-through carried through

    def test_empty_rows_returns_none(self):
        self.assertIsNone(build_poses_table([], []))


class PosesToChemMolsTest(WidgetTest):
    def test_pose_carries_docked_conformer_and_data(self):
        block = molecule_to_sdf_record(_mol_3d(), "lig")
        row = {"status": "ok", "pose_sdf": block, "ligand_name": "lig",
               "rank_by_score": 1, "pose_index": 1, "score": -5.0,
               "score_field": "SCORE.INTER", "_properties": {"SCORE": "-9.0"},
               "compound_id": "CMPD-1"}
        mols = poses_to_chemmols([row], ["compound_id"])
        self.assertEqual(len(mols), 1)
        cm = mols[0]
        self.assertTrue(cm.mol.GetConformer().Is3D())     # docked 3D geometry
        self.assertEqual(cm.props["compound_id"], "CMPD-1")  # original data
        self.assertEqual(cm.props["score"], -5.0)            # docking data
        self.assertEqual(cm.props["SCORE"], "-9.0")

    def test_failed_rows_are_skipped(self):
        self.assertEqual(poses_to_chemmols([{"status": "failed", "pose_sdf": ""}], []), [])


class OWCmDockDockingTest(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCmDockDocking)

    @staticmethod
    def _ligand_table() -> Table:
        domain = Domain([], metas=[StringVariable("sdf"), StringVariable("compound_id")])
        metas = np.array([["LigandA\n$$$$\n", "CMPD-1"]], dtype=object)
        return Table.from_numpy(domain, X=np.empty((1, 0)), metas=metas)

    def test_set_data_populates_sdf_column(self):
        self.widget.set_data(self._ligand_table())
        self.assertEqual(self.widget.sdf_column, "sdf")
        self.assertEqual(self.widget._column_combo.count(), 2)

    def test_frame_from_table_passes_through_non_sdf_columns(self):
        self.widget.set_data(self._ligand_table())
        self.widget.sdf_column = "sdf"
        frame, sdf_col, passthrough_map, names = self.widget._frame_from_table()
        self.assertEqual(sdf_col, "sdf")
        self.assertEqual(passthrough_map, {"compound_id": "compound_id"})
        self.assertEqual(names, ["compound_id"])

    def test_frame_from_molecules_uses_props_when_no_data(self):
        cm = ChemMol(mol=_mol_3d(), name="lig", props={"compound_id": "CMPD-1"})
        self.widget.set_molecules([cm])
        frame, sdf_col, passthrough_map, names = self.widget._frame_from_molecules()
        self.assertEqual(sdf_col, "__sdf__")
        self.assertIn("__sdf__", frame.columns)
        self.assertIn("$$$$", frame["__sdf__"].iloc[0])
        self.assertIn("compound_id", names)

    @staticmethod
    def _make_prm(with_cavity: bool, receptor_file: "str | None" = None) -> str:
        d = tempfile.mkdtemp()
        prm = os.path.join(d, "rec.prm")
        with open(prm, "w") as f:
            f.write("RBT_PARAMETER_FILE_V1.00\n")
            if receptor_file:
                f.write(f"RECEPTOR_FILE {receptor_file}\n")
        if with_cavity:
            open(os.path.join(d, "rec.as"), "w").close()
        return prm

    def test_missing_receptor_file_sets_error(self):
        self.widget.set_molecules([ChemMol(mol=_mol_3d(), name="lig")])
        # cavity present, but RECEPTOR_FILE points at a file that does not exist
        self.widget.receptor_prm = self._make_prm(with_cavity=True, receptor_file="target.mol2")
        self.widget._start_docking()
        self.assertTrue(self.widget.Error.active)

    def test_dock_without_receptor_sets_error(self):
        self.widget.set_molecules([ChemMol(mol=_mol_3d(), name="lig")])
        self.widget.receptor_prm = ""
        self.widget._start_docking()
        self.assertTrue(self.widget.Error.active)

    def test_missing_cavity_sets_error(self):
        self.widget.set_molecules([ChemMol(mol=_mol_3d(), name="lig")])
        self.widget.receptor_prm = self._make_prm(with_cavity=False)
        self.widget._start_docking()
        self.assertTrue(self.widget.Error.active)

    def test_no_inputs_with_valid_receptor_sets_error(self):
        self.widget.receptor_prm = self._make_prm(with_cavity=True)
        self.widget._start_docking()  # passes preflight, fails on missing inputs
        self.assertTrue(self.widget.Error.active)


if __name__ == "__main__":
    unittest.main(verbosity=2)
