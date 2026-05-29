from __future__ import annotations

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")
pytest.importorskip("rdkit")

from orangecanvas.registry import WidgetRegistry
from orangewidget.workflow.discovery import WidgetDiscovery

import chem_inf_widgets.widgets as widget_package
from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.molecule_export_service import (
    MoleculeExportConfig,
    detect_export_format,
    export_molecule_data,
)
from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table


def test_detect_export_format():
    assert detect_export_format("demo.csv") == "csv"
    assert detect_export_format("demo.tsv") == "tsv"
    assert detect_export_format("demo.txt") == "txt"
    assert detect_export_format("demo.smi") == "smi"
    assert detect_export_format("demo.sdf") == "sdf"


def test_export_molecules_to_smiles_file(tmp_path):
    output_path = tmp_path / "demo.smi"
    molecules = [
        ChemMol.from_smiles("CCO", name="ethanol"),
        ChemMol.from_smiles("c1ccncc1", name="pyridine"),
    ]

    result = export_molecule_data(
        output_path,
        molecules=molecules,
        config=MoleculeExportConfig(output_format="smi", include_header=True, include_props=False),
    )

    text = output_path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "SMILES\tName"
    assert "CCO\tethanol" in text
    assert "c1ccncc1\tpyridine" in text
    assert result.summary.output_format == "smi"
    assert result.summary.total_records == 2
    assert result.summary.written_records == 2
    assert result.summary.columns == ["SMILES", "Name"]


def test_export_data_to_csv_reports_failed_rows(tmp_path):
    output_path = tmp_path / "demo.csv"
    table = records_to_orange_table(
        [
            {"compound_id": "M001", "SMILES": "CCO", "activity": 1.2},
            {"compound_id": "M002", "SMILES": "C1CC", "activity": 2.3},
            {"compound_id": "M003", "SMILES": "c1ccncc1", "activity": 3.4},
        ],
        name="Export Demo",
    )

    result = export_molecule_data(
        output_path,
        data=table,
        config=MoleculeExportConfig(
            output_format="csv",
            smiles_column="SMILES",
            name_column="compound_id",
            include_props=False,
            include_header=True,
        ),
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "SMILES,Name"
    assert "CCO,M001" in lines
    assert "c1ccncc1,M003" in lines
    assert all("M002" not in line for line in lines)
    assert result.summary.total_records == 3
    assert result.summary.valid_records == 2
    assert result.summary.failed_records == 1
    assert result.summary.written_records == 2
    assert result.summary.columns == ["SMILES", "Name"]
    assert any("SMILES" in record.error or record.error for record in result.failed_records)


def test_widget_discovery_registers_molecule_export_hub():
    registry = WidgetRegistry()
    discovery = WidgetDiscovery(registry)

    widget_package.widget_discovery(discovery)

    assert registry.has_widget(
        "chem_inf_widgets.widgets.ow_molecule_export_hub.OWMoleculeExportHub"
    )
