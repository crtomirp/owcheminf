from __future__ import annotations

import csv
from pathlib import Path

from chem_inf_widgets.chemcore.tools.cyclic_registry_fingerprint_cli import (
    build_parser,
    read_input,
)


def test_cyclic_registry_cli_parser_knows_main_options():
    parser = build_parser()
    args = parser.parse_args(["input.csv", "--smiles-column", "SMILES", "--name-column", "Name", "--write-json"])
    assert args.input == Path("input.csv")
    assert args.smiles_column == "SMILES"
    assert args.name_column == "Name"
    assert args.write_json is True


def test_cyclic_registry_cli_reads_csv(tmp_path):
    p = tmp_path / "mols.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "smiles", "class"])
        writer.writeheader()
        writer.writerow({"name": "pyridine", "smiles": "c1ccncc1", "class": "heterocycle"})
    args = build_parser().parse_args([str(p), "--smiles-column", "smiles", "--name-column", "name"])
    ids, names, smiles = read_input(args)
    assert ids == ["1"]
    assert names == ["pyridine"]
    assert smiles == ["c1ccncc1"]


def test_cyclic_registry_cli_reads_smi(tmp_path):
    p = tmp_path / "mols.smi"
    p.write_text("c1ccncc1 pyridine\n# comment\n\nc1ccoc1 furan\n", encoding="utf-8")
    args = build_parser().parse_args([str(p)])
    ids, names, smiles = read_input(args)
    assert ids == ["1", "2"]
    assert names == ["pyridine", "furan"]
    assert smiles == ["c1ccncc1", "c1ccoc1"]
