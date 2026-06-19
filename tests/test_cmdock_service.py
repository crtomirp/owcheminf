"""Tests for the GUI-free CmDock docking service (no Orange, no real CmDock).

A fake ``cmdock`` executable writes a canned SDF to its ``-o`` target, mirroring
the KNIME reference tests, so the orchestration can be exercised deterministically.
"""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("pandas required") from exc

from chem_inf_widgets.chemcore.services.cmdock_service import (
    DockingSettings,
    dock_dataframe,
    parse_sdf_records,
    read_ligand_input,
    referenced_receptor_file,
)

SDF = (
    "LigandA\n  Test\n\n  0  0  0  0  0  0            999 V2000\nM  END\n"
    "> <SCORE.INTER>\n-7.5\n\n> <SCORE>\n-9.1\n\n> <NAME>\naspirin\n\n$$$$\n"
)


def _fake_cmdock(tmp: Path, sdf: str = SDF, return_code: int = 0) -> Path:
    exe = tmp / "fake_cmdock.py"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.exit({return_code}) if {return_code} else None\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        f"open(out, 'w').write({sdf!r})\n",
        encoding="utf-8",
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def _settings(exe: Path, prm: Path, **kw) -> DockingSettings:
    return DockingSettings(
        cmdock_executable=str(exe), receptor_prm=str(prm),
        n_docking_runs=1, n_best_poses=1, score_tag="SCORE.INTER", **kw,
    )


class CmDockServiceTest(unittest.TestCase):
    def test_parse_sdf_records(self):
        records = parse_sdf_records(SDF)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].name, "LigandA")
        self.assertEqual(records[0].properties["SCORE.INTER"], "-7.5")
        self.assertIn("$$$$", records[0].sdf_text)

    def test_dock_dataframe_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            exe, prm = _fake_cmdock(tmp), tmp / "receptor.prm"
            prm.write_text("RBT_PARAMETER_FILE_V1.00\n", encoding="utf-8")
            df = pd.DataFrame({"compound_id": ["CMPD-1"], "sdf": [SDF]})

            rows = dock_dataframe(df, "sdf", _settings(exe, prm),
                                  passthrough_column_map={"compound_id": "compound_id"})

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["compound_id"], "CMPD-1")
            self.assertAlmostEqual(row["score"], -7.5)
            self.assertEqual(row["rank_by_score"], 1)
            self.assertEqual(row["_properties"]["NAME"], "aspirin")

    def test_failed_ligand_produces_error_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            exe = _fake_cmdock(tmp, return_code=3)
            prm = tmp / "receptor.prm"
            prm.write_text("RBT_PARAMETER_FILE_V1.00\n", encoding="utf-8")
            df = pd.DataFrame({"sdf": [SDF]})

            rows = dock_dataframe(df, "sdf", _settings(exe, prm))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIn("return code 3", rows[0]["error_message"])

    def test_read_ligand_input_handles_long_multiline_sdf(self):
        # Regression: a long multi-line SDF must not be probed as a filesystem
        # path (Path.is_file() raises ENAMETOOLONG on >255-char components).
        big = "MOL\n" + "C" * 600 + "\nM  END\n$$$$\n"
        self.assertEqual(read_ligand_input(big, "Auto"), big)

    def test_referenced_receptor_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            prm = Path(tmp) / "rec.prm"
            prm.write_text("RBT_PARAMETER_FILE_V1.00\nRECEPTOR_FILE target.mol2\n", encoding="utf-8")
            self.assertEqual(referenced_receptor_file(prm), "target.mol2")
            prm.write_text("RBT_PARAMETER_FILE_V1.00\n", encoding="utf-8")
            self.assertIsNone(referenced_receptor_file(prm))

    def test_missing_sdf_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            exe, prm = _fake_cmdock(tmp), tmp / "receptor.prm"
            prm.write_text("x\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                dock_dataframe(pd.DataFrame({"sdf": [SDF]}), "missing", _settings(exe, prm))


if __name__ == "__main__":
    unittest.main(verbosity=2)
