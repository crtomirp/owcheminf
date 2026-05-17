from __future__ import annotations

import pytest

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")
pytest.importorskip("rdkit")

from AnyQt.QtWidgets import QApplication
from orangecanvas.registry import WidgetRegistry
from orangewidget.workflow.discovery import WidgetDiscovery

import chem_inf_widgets.widgets as widget_package
from chem_inf_widgets.widgets import _widget_desc_from_local_module
from chem_inf_widgets.widgets import ow_widget_smoke_tester as smoke_tester_module
from chem_inf_widgets.widgets.ow_widget_smoke_tester import (
    CORE_WORKFLOW_MODULES,
    CORE_WORKFLOW_PROVENANCE_COLUMNS,
    OWWidgetSmokeTester,
    discover_widget_smoke_targets,
    run_core_workflow_smoke,
    run_workflow_smoke_suite,
    run_widget_smoke_checks,
    smoke_records_to_table,
    smoke_summary_to_table,
    workflow_records_to_table,
    workflow_summary_to_table,
)


_APP = QApplication.instance() or QApplication([])


def test_discover_widget_smoke_targets_includes_core_workflow():
    targets = discover_widget_smoke_targets("core-workflow")
    module_names = {target.module_name for target in targets}

    for module_name in CORE_WORKFLOW_MODULES:
        assert module_name in module_names


def test_widget_discovery_uses_local_widget_class_for_smoke_tester_module():
    desc = _widget_desc_from_local_module(smoke_tester_module)

    assert desc.qualified_name == (
        "chem_inf_widgets.widgets.ow_widget_smoke_tester.OWWidgetSmokeTester"
    )


def test_widget_package_discovery_registers_without_duplicate_widgets():
    registry = WidgetRegistry()
    discovery = WidgetDiscovery(registry)

    widget_package.widget_discovery(discovery)

    assert registry.has_widget(
        "chem_inf_widgets.widgets.ow_widget_smoke_tester.OWWidgetSmokeTester"
    )
    assert registry.has_widget(
        "chem_inf_widgets.widgets.ow_mol_standardizer.OWMolStandardizer"
    )


def test_widget_smoke_runner_imports_and_instantiates_core_subset():
    subset = [
        "ow_widget_smoke_tester",
        "ow_mol_editor",
        "ow_molecule_import_hub",
        "ow_molecule_qc_dashboard",
        "ow_mol_standardizer",
        "ow_qsar_dataset_builder",
    ]
    records, summary = run_widget_smoke_checks(
        scope="all",
        instantiate_widgets=True,
        modules=subset,
    )

    assert len(records) == len(subset)
    assert summary["failed"] == 0
    assert summary["import_ok"] == len(subset)
    assert summary["instantiate_ok"] == len(subset)
    assert all(row["status"] == "ok" for row in records)

    report_table = smoke_records_to_table(records)
    summary_table = smoke_summary_to_table(summary)
    assert report_table is not None
    assert summary_table is not None
    assert len(report_table) == len(subset)
    assert len(summary_table) >= 1


def test_widget_smoke_tester_widget_runs_and_outputs_tables():
    widget = OWWidgetSmokeTester()
    try:
        widget.scope = "core-workflow"
        widget.instantiate_widgets = False
        widget._run()

        assert widget.results_table.rowCount() >= len(CORE_WORKFLOW_MODULES)
        assert "Smoke check complete" in widget.lbl.text()
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_core_workflow_smoke_runs_end_to_end_and_preserves_provenance():
    records, summary, ready_table = run_core_workflow_smoke()

    assert len(records) == 5
    assert summary["failed"] == 0
    assert summary["workflow_ok"] == 1
    assert summary["prepared_compounds"] >= 1
    assert ready_table is not None
    assert records[0]["stage"] == "mol_editor"
    assert records[0]["widget_class"] == "OWMolSketcher"
    ready_meta_names = {var.name for var in ready_table.domain.metas}
    for name in CORE_WORKFLOW_PROVENANCE_COLUMNS:
        assert name in ready_meta_names

    report_table = workflow_records_to_table(records)
    summary_table = workflow_summary_to_table(summary)
    assert report_table is not None
    assert summary_table is not None
    assert len(report_table) == 5


def test_widget_smoke_tester_widget_runs_core_workflow_smoke():
    widget = OWWidgetSmokeTester()
    try:
        widget._run_workflow_smoke()

        assert widget.results_table.rowCount() == 5
        assert "Workflow smoke complete" in widget.lbl.text()
        assert widget.results_table.item(0, 0).text() == "mol_editor"
        assert widget.results_table.item(4, 0).text() == "qsar_builder"
        assert widget.results_table.item(4, 2).text() == "ok"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_workflow_smoke_suite_runs_extended_paths():
    records, summary, ready_table = run_workflow_smoke_suite()

    workflows = {row["workflow"] for row in records}
    assert workflows == {"core_workflow", "descriptor_model_validation", "chembl_builder"}
    assert summary["workflow_failures"] == 0
    assert summary["workflow_ok"] == 1
    assert summary["prepared_compounds"] >= 2
    assert summary["validation_outliers"] >= 0
    assert ready_table is not None

    report_table = workflow_records_to_table(records)
    summary_table = workflow_summary_to_table(summary)
    assert report_table is not None
    assert summary_table is not None
    assert len(report_table) >= 10


def test_widget_smoke_tester_widget_runs_workflow_suite():
    widget = OWWidgetSmokeTester()
    try:
        widget._run_workflow_suite()

        assert widget.results_table.rowCount() >= 10
        assert "Workflow suite complete" in widget.lbl.text()
        assert widget.results_table.item(0, 0).text() in {
            "core_workflow",
            "descriptor_model_validation",
            "chembl_builder",
        }
    finally:
        widget.onDeleteWidget()
        widget.close()
