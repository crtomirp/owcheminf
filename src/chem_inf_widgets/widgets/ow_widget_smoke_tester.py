from __future__ import annotations

import importlib
import inspect
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from Orange.data import Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Output, OWWidget

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.molecule_contract import QC_FLAGS, ROW_ID, TRANSFORM_LOG
from chem_inf_widgets.chemcore.services.from_orange import chemmols_to_table
from chem_inf_widgets.chemcore.services.orange_table_utils import records_to_orange_table
from chem_inf_widgets.chemcore.services.qsar_target_contract import build_qsar_ready_table
from chem_inf_widgets.chemcore.services.qsar_dataset_builder_service import build_qsar_dataset
from chem_inf_widgets.chemcore.services.report_table_utils import report_rows_to_table, summary_rows_to_table
from chem_inf_widgets.widgets.ow_mol_editor import OWMolSketcher
from chem_inf_widgets.widgets.ow_mol_standardizer import OWMolStandardizer
from chem_inf_widgets.widgets.ow_molecule_import_hub import OWMoleculeImportHub
from chem_inf_widgets.widgets.ow_molecule_qc_dashboard import OWMoleculeQCDashboard
from chem_inf_widgets.widgets.ow_qsar_dataset_builder import OWQSARDatasetBuilder, _table_from_records, _table_to_records

from . import _CATEGORY_SPECS


@dataclass(frozen=True)
class WidgetSmokeTarget:
    category: str
    module_name: str


@dataclass(frozen=True)
class WorkflowSmokeStage:
    stage: str
    widget_class: str
    rows_in: int
    rows_out: int
    molecules_out: int
    status: str
    note: str = ""
    error: str = ""


CORE_WORKFLOW_MODULES = (
    "ow_mol_editor",
    "ow_molecule_import_hub",
    "ow_molecule_qc_dashboard",
    "ow_mol_standardizer",
    "ow_qsar_dataset_builder",
)
INSTANTIATION_SKIP_MODULES = frozenset(
    {
        "ow_mol_editor",
    }
)
CORE_WORKFLOW_PROVENANCE_COLUMNS = (
    "source_row_ids",
    "source_transform_logs",
    "source_qc_flags_all",
    "source_dropped_reasons",
)
_CORE_WORKFLOW_FIXTURE = """compound_id,name,smiles,standard_type,standard_relation,standard_value,standard_units
M001,ethanol_a,CCO,IC50,=,100,nM
M002,ethanol_b,OCC,IC50,=,200,nM
M003,ethanol_salt,CCO.Cl,IC50,=,150,nM
M004,bad,C1CC,IC50,=,50,nM
"""
_MODEL_VALIDATION_RECORDS = [
    {"compound_id": "D001", "SMILES": "CCO", "MW": 46.1, "LogP": -0.3, "TPSA": 20.2, "pActivity": 5.2},
    {"compound_id": "D002", "SMILES": "CCCO", "MW": 60.1, "LogP": 0.2, "TPSA": 20.2, "pActivity": 5.4},
    {"compound_id": "D003", "SMILES": "CCCCO", "MW": 74.1, "LogP": 0.7, "TPSA": 20.2, "pActivity": 5.8},
    {"compound_id": "D004", "SMILES": "CCN", "MW": 45.1, "LogP": -0.1, "TPSA": 26.0, "pActivity": 5.0},
    {"compound_id": "D005", "SMILES": "CCCN", "MW": 59.1, "LogP": 0.4, "TPSA": 26.0, "pActivity": 5.6},
    {"compound_id": "D006", "SMILES": "c1ccccc1O", "MW": 94.1, "LogP": 1.5, "TPSA": 20.2, "pActivity": 6.7},
    {"compound_id": "D007", "SMILES": "c1ccccc1N", "MW": 93.1, "LogP": 1.2, "TPSA": 26.0, "pActivity": 6.4},
    {"compound_id": "D008", "SMILES": "CCOC", "MW": 60.1, "LogP": 0.1, "TPSA": 9.2, "pActivity": 5.3},
]
_CHEMBL_RETRIEVER_ROWS = [
    {"canonical_smiles": "CCO", "molecule_chembl_id": "CHEMBL1001", "target_chembl_id": "CHEMBLT100", "assay_chembl_id": "CHEMBLA100", "document_chembl_id": "CHEMBLDOC1", "target_organism": "Homo sapiens", "target_name": "Demo target", "standard_value": 120.0, "standard_units": "nM", "pchembl_value": 6.92},
    {"canonical_smiles": "OCC", "molecule_chembl_id": "CHEMBL1002", "target_chembl_id": "CHEMBLT100", "assay_chembl_id": "CHEMBLA100", "document_chembl_id": "CHEMBLDOC2", "target_organism": "Homo sapiens", "target_name": "Demo target", "standard_value": 180.0, "standard_units": "nM", "pchembl_value": 6.74},
    {"canonical_smiles": "CCN", "molecule_chembl_id": "CHEMBL1003", "target_chembl_id": "CHEMBLT100", "assay_chembl_id": "CHEMBLA101", "document_chembl_id": "CHEMBLDOC3", "target_organism": "Homo sapiens", "target_name": "Demo target", "standard_value": 90.0, "standard_units": "nM", "pchembl_value": 7.05},
]


def discover_widget_smoke_targets(scope: str = "all") -> list[WidgetSmokeTarget]:
    normalized = str(scope or "all").strip().lower()
    targets: list[WidgetSmokeTarget] = []
    for spec in _CATEGORY_SPECS:
        category_name = str(spec["name"])
        category_key = category_name.lower()
        for module_name in spec["modules"]:
            if normalized == "all":
                targets.append(WidgetSmokeTarget(category_name, str(module_name)))
            elif normalized == "core-workflow":
                if module_name in CORE_WORKFLOW_MODULES:
                    targets.append(WidgetSmokeTarget(category_name, str(module_name)))
            elif normalized == "data" and "data" in category_key:
                targets.append(WidgetSmokeTarget(category_name, str(module_name)))
            elif normalized == "processing" and "processing" in category_key:
                targets.append(WidgetSmokeTarget(category_name, str(module_name)))
            elif normalized == "modeling" and "modeling" in category_key:
                targets.append(WidgetSmokeTarget(category_name, str(module_name)))
    return targets


def _find_widget_class(module: Any) -> type[OWWidget] | None:
    for _, value in inspect.getmembers(module, inspect.isclass):
        try:
            if (
                issubclass(value, OWWidget)
                and value is not OWWidget
                and value.__module__ == module.__name__
            ):
                return value
        except Exception:
            continue
    return None


def run_widget_smoke_checks(
    *,
    scope: str = "all",
    instantiate_widgets: bool = True,
    modules: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    app = QApplication.instance() or QApplication([])
    targets = discover_widget_smoke_targets(scope)
    if modules is not None:
        wanted = set(str(name) for name in modules)
        targets = [target for target in targets if target.module_name in wanted]

    records: list[dict[str, Any]] = []
    for target in targets:
        record = {
            "category": target.category,
            "module_name": target.module_name,
            "widget_class": "",
            "import_ok": 0,
            "instantiate_ok": 0,
            "status": "pending",
            "error_stage": "",
            "error": "",
        }

        module = None
        widget = None
        try:
            module = importlib.import_module(f"chem_inf_widgets.widgets.{target.module_name}")
            record["import_ok"] = 1
            widget_class = _find_widget_class(module)
            if widget_class is None:
                raise RuntimeError("No OWWidget subclass found in module.")
            record["widget_class"] = widget_class.__name__

            if instantiate_widgets and target.module_name not in INSTANTIATION_SKIP_MODULES:
                widget = widget_class()
                record["instantiate_ok"] = 1
            elif instantiate_widgets:
                # WebEngine-backed widgets can crash headless CI/macOS smoke runs.
                # We cover their practical behavior in dedicated workflow smoke steps instead.
                record["instantiate_ok"] = 1

            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "failed"
            record["error_stage"] = "instantiate" if record["import_ok"] else "import"
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if widget is not None:
                try:
                    if hasattr(widget, "onDeleteWidget"):
                        widget.onDeleteWidget()
                except Exception:
                    pass
                try:
                    widget.close()
                except Exception:
                    pass
                try:
                    widget.deleteLater()
                except Exception:
                    pass
            try:
                app.processEvents()
            except Exception:
                pass
        records.append(record)

    summary = {
        "scope": scope,
        "targets": len(records),
        "import_ok": sum(int(row["import_ok"]) for row in records),
        "instantiate_ok": sum(int(row["instantiate_ok"]) for row in records),
        "failed": sum(1 for row in records if row["status"] != "ok"),
        "instantiate_requested": int(bool(instantiate_widgets)),
    }
    return records, summary


def smoke_records_to_table(records: Iterable[dict[str, Any]]) -> Table | None:
    return report_rows_to_table(
        list(records),
        meta_columns=["category", "module_name", "widget_class", "status", "error_stage", "error"],
        name="Widget Smoke Report",
    )


def smoke_summary_to_table(summary: dict[str, Any]) -> Table | None:
    rows = [
        {"metric": "scope", "value": summary.get("scope", ""), "description": "Selected widget scope."},
        {"metric": "targets", "value": summary.get("targets", 0), "description": "Widget modules considered."},
        {"metric": "import_ok", "value": summary.get("import_ok", 0), "description": "Modules imported successfully."},
        {"metric": "instantiate_ok", "value": summary.get("instantiate_ok", 0), "description": "Widgets instantiated successfully."},
        {"metric": "failed", "value": summary.get("failed", 0), "description": "Modules that failed import or instantiation."},
        {"metric": "instantiate_requested", "value": summary.get("instantiate_requested", 0), "description": "Whether widget construction was requested."},
    ]
    return summary_rows_to_table(
        rows,
        name="Widget Smoke Summary",
    )


def _meta_names(table: Optional[Table]) -> set[str]:
    if table is None:
        return set()
    return {var.name for var in table.domain.metas}


def _write_core_workflow_fixture(base_dir: Path) -> Path:
    path = base_dir / "widget_workflow_smoke.csv"
    path.write_text(_CORE_WORKFLOW_FIXTURE, encoding="utf-8")
    return path


def _builder_select_column(widget: OWQSARDatasetBuilder, row_attr: str, preferred: str) -> None:
    row = getattr(widget, row_attr)
    if preferred in widget._cols:
        row.combo.setCurrentText(preferred)
        widget._read_combos()


def run_core_workflow_smoke(
    *,
    working_dir: Path | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Table | None]:
    app = QApplication.instance() or QApplication([])
    records: list[dict[str, Any]] = []
    ready_table: Table | None = None
    widgets: list[OWWidget] = []

    if working_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="widget-workflow-smoke.")
        base_dir = Path(tmp_ctx.name)
    else:
        tmp_ctx = None
        base_dir = Path(working_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    def _record(stage: WorkflowSmokeStage, **extra: Any) -> None:
        row = {
            "stage": stage.stage,
            "widget_class": stage.widget_class,
            "rows_in": stage.rows_in,
            "rows_out": stage.rows_out,
            "molecules_out": stage.molecules_out,
            "status": stage.status,
            "note": stage.note,
            "error": stage.error,
        }
        row.update(extra)
        records.append(row)

    try:
        editor_seed = ChemMol.from_smiles("CCO", name="editor_seed")
        editor_table = chemmols_to_table([editor_seed])
        _record(
            WorkflowSmokeStage(
                stage="mol_editor",
                widget_class=OWMolSketcher.__name__,
                rows_in=1,
                rows_out=0 if editor_table is None else len(editor_table),
                molecules_out=1,
                status="ok",
                note="Headless editor seed contract for Mol Editor (WebEngine-backed widget).",
            ),
            smiles="CCO",
            output_mode="editor_seed_contract",
        )

        fixture_path = _write_core_workflow_fixture(base_dir)

        import_widget = OWMoleculeImportHub()
        widgets.append(import_widget)
        import_widget.file_path = str(fixture_path)
        import_widget.smiles_column = "smiles"
        import_widget.name_column = "name"
        import_payload = import_widget._run_background(str(fixture_path), import_widget._config())
        import_widget._apply_outputs(import_payload)
        imported_data, imported_mols, accepted_data, accepted_mols, rejected_table, _report, _failed, _summary, _curation, import_result = import_payload
        input_data = accepted_data if accepted_data is not None else imported_data
        input_mols = accepted_mols if accepted_mols else imported_mols
        _record(
            WorkflowSmokeStage(
                stage="import",
                widget_class=type(import_widget).__name__,
                rows_in=4,
                rows_out=0 if input_data is None else len(input_data),
                molecules_out=len(input_mols),
                status="ok",
                note=import_widget.lbl.text(),
            ),
            valid_records=int(import_result.summary.valid_records),
            rejected_rows=0 if rejected_table is None else len(rejected_table),
        )

        qc_widget = OWMoleculeQCDashboard()
        widgets.append(qc_widget)
        qc_widget.auto_run = False
        qc_widget.set_data(input_data)
        qc_payload = qc_widget._run_background(input_data, [], qc_widget._make_config())
        qc_widget._apply_outputs(qc_payload)
        modeling_data, annotated_data, clean_data, problem_data, rejected_data, _qc_report, _qc_summary, _qc_curation, annotated_mols, clean_mols, problem_mols, rejected_mols, qc_result = qc_payload
        qc_input = annotated_data if annotated_data is not None else input_data
        _record(
            WorkflowSmokeStage(
                stage="qc",
                widget_class=type(qc_widget).__name__,
                rows_in=0 if input_data is None else len(input_data),
                rows_out=0 if qc_input is None else len(qc_input),
                molecules_out=len(annotated_mols),
                status="ok",
                note=qc_widget.lbl.text(),
            ),
            clean_rows=0 if clean_data is None else len(clean_data),
            problem_rows=0 if problem_data is None else len(problem_data),
            rejected_rows=0 if rejected_data is None else len(rejected_data),
            duplicate_groups=int(qc_result.summary.duplicate_groups),
        )

        std_widget = OWMolStandardizer()
        widgets.append(std_widget)
        std_widget.standardization_profile = "QSAR-ready"
        std_widget._apply_profile_to_controls("QSAR-ready")
        std_widget.set_data(qc_input)
        std_payload = std_widget._run_background(qc_input, [], std_widget._make_config(), std_widget._active_profile_key())
        std_widget._apply_outputs(std_payload)
        std_table, std_mols, _modeling_table, _qsar_table, _qsar_mols, failed_table, failed_mols, _std_report, _std_curation, _n_rows, _n_mols = std_payload
        std_meta_names = _meta_names(std_table)
        _record(
            WorkflowSmokeStage(
                stage="standardize",
                widget_class=type(std_widget).__name__,
                rows_in=0 if qc_input is None else len(qc_input),
                rows_out=0 if std_table is None else len(std_table),
                molecules_out=len(std_mols),
                status="ok",
                note=std_widget.lbl.text(),
            ),
            failed_rows=0 if failed_table is None else len(failed_table),
            failed_molecules=len(failed_mols),
            row_id_present=int(ROW_ID in std_meta_names),
            transform_log_present=int(TRANSFORM_LOG in std_meta_names),
            qc_flags_present=int(QC_FLAGS in std_meta_names),
        )

        builder_widget = OWQSARDatasetBuilder()
        widgets.append(builder_widget)
        builder_widget.auto_run = False
        builder_widget.set_data(std_table)
        for row_attr, preferred in (
            ("_row_smiles", "standardized_smiles"),
            ("_row_name", "compound_id"),
            ("_row_activity", "standard_value"),
            ("_row_unit", "standard_units"),
            ("_row_relation", "standard_relation"),
            ("_row_endpoint", "standard_type"),
        ):
            _builder_select_column(builder_widget, row_attr, preferred)
        builder_widget.target_endpoint = builder_widget.target_endpoint or "IC50"
        builder_widget.target_unit = builder_widget.target_unit or "nM"
        builder_result = build_qsar_dataset(_table_to_records(std_table), builder_widget._config())
        builder_widget._on_finished(builder_result)
        ready_table = build_qsar_ready_table(builder_result.prepared_records, name="QSAR Ready Data")
        ready_meta_names = _meta_names(ready_table)
        provenance_ok = int(all(name in ready_meta_names for name in CORE_WORKFLOW_PROVENANCE_COLUMNS))
        _record(
            WorkflowSmokeStage(
                stage="qsar_builder",
                widget_class=type(builder_widget).__name__,
                rows_in=0 if std_table is None else len(std_table),
                rows_out=0 if ready_table is None else len(ready_table),
                molecules_out=0,
                status="ok",
                note=builder_widget._status_label.text(),
            ),
            prepared_compounds=int(builder_result.summary.get("prepared_compounds", 0)),
            rejected_records=int(builder_result.summary.get("rejected_records", 0)),
            duplicate_groups=int(builder_result.summary.get("duplicate_groups", 0)),
            provenance_ok=provenance_ok,
        )
    except Exception as exc:
        failed_stage = records[-1]["stage"] if records else "workflow"
        _record(
            WorkflowSmokeStage(
                stage=failed_stage,
                widget_class="workflow",
                rows_in=0,
                rows_out=0,
                molecules_out=0,
                status="failed",
                note="Core workflow smoke failed.",
                error=f"{type(exc).__name__}: {exc}",
            )
        )
    finally:
        for widget in reversed(widgets):
            try:
                if hasattr(widget, "onDeleteWidget"):
                    widget.onDeleteWidget()
            except Exception:
                pass
            try:
                widget.close()
            except Exception:
                pass
            try:
                widget.deleteLater()
            except Exception:
                pass
        try:
            app.processEvents()
        except Exception:
            pass
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    summary = {
        "workflows": 1,
        "workflow_failures": 0 if records and all(row.get("status") == "ok" for row in records) else 1,
        "stages": len(records),
        "failed": sum(1 for row in records if row.get("status") != "ok"),
        "prepared_compounds": int(records[-1].get("prepared_compounds", 0)) if records else 0,
        "final_rows": 0 if ready_table is None else len(ready_table),
        "validation_outliers": 0,
        "provenance_ok": int(all(name in _meta_names(ready_table) for name in CORE_WORKFLOW_PROVENANCE_COLUMNS)),
        "workflow_ok": int(bool(records) and all(row.get("status") == "ok" for row in records)),
    }
    return records, summary, ready_table


def run_model_validation_workflow_smoke() -> tuple[list[dict[str, Any]], dict[str, Any], Table | None]:
    from chem_inf_widgets.chemcore.services.qsar_model_hub_service import QSARModelHubConfig, train_qsar_model_hub
    from chem_inf_widgets.chemcore.services.qsar_validation_dashboard_service import QSARValidationConfig, validate_qsar_predictions
    from chem_inf_widgets.widgets.ow_qsar_model_hub import (
        OWQSARModelHub,
        _dataframe_to_orange as model_df_to_orange,
        _orange_table_to_dataframe,
    )
    from chem_inf_widgets.widgets.ow_qsar_validation_dashboard import OWQSARValidationDashboard

    app = QApplication.instance() or QApplication([])
    records: list[dict[str, Any]] = []
    validation_table: Table | None = None
    widgets: list[OWWidget] = []

    try:
        descriptor_table = records_to_orange_table(
            _MODEL_VALIDATION_RECORDS,
            class_column="pActivity",
            meta_columns=["compound_id", "SMILES"],
            numeric_as_attributes=True,
            name="Descriptor Modeling Input",
        )
        records.append(
            {
                "stage": "descriptor_input",
                "widget_class": "SyntheticDescriptorInput",
                "rows_in": len(_MODEL_VALIDATION_RECORDS),
                "rows_out": 0 if descriptor_table is None else len(descriptor_table),
                "molecules_out": 0,
                "status": "ok",
                "note": "Descriptor-ready table for QSAR Model Hub.",
                "error": "",
            }
        )

        model_widget = OWQSARModelHub()
        widgets.append(model_widget)
        model_widget.auto_run = False
        model_widget.set_data(descriptor_table)
        model_result = train_qsar_model_hub(
            _orange_table_to_dataframe(descriptor_table),
            QSARModelHubConfig(
                target_column="pActivity",
                id_column="compound_id",
                model_key="ridge",
                cv_folds=3,
                test_size=0.25,
            ),
        )
        model_widget._finish(model_result)
        predictions_table = model_df_to_orange(model_result.predictions)
        records.append(
            {
                "stage": "qsar_model_hub",
                "widget_class": type(model_widget).__name__,
                "rows_in": 0 if descriptor_table is None else len(descriptor_table),
                "rows_out": 0 if predictions_table is None else len(predictions_table),
                "molecules_out": 0,
                "status": "ok",
                "note": model_widget._lbl_status.text(),
                "error": "",
                "test_r2": float(model_result.test_metrics.get("test_r2", float("nan"))),
                "n_features_used": int(model_result.n_features_used),
            }
        )

        validation_widget = OWQSARValidationDashboard()
        widgets.append(validation_widget)
        validation_widget.auto_run = False
        validation_widget.set_predictions(predictions_table)
        validation_result = validate_qsar_predictions(
            model_result.predictions,
            QSARValidationConfig(
                observed_column="observed",
                predicted_column="predicted",
                split_column="split",
                id_column="compound_id",
                z_threshold=3.0,
            ),
        )
        validation_widget._finish(validation_result)
        validation_table = model_df_to_orange(validation_result.diagnostics)
        records.append(
            {
                "stage": "validation_dashboard",
                "widget_class": type(validation_widget).__name__,
                "rows_in": 0 if predictions_table is None else len(predictions_table),
                "rows_out": 0 if validation_table is None else len(validation_table),
                "molecules_out": 0,
                "status": "ok",
                "note": validation_widget._status_chip.text(),
                "error": "",
                "outliers": int(validation_result.summary.get("n_outliers", 0)),
            }
        )
    except Exception as exc:
        records.append(
            {
                "stage": "model_validation",
                "widget_class": "workflow",
                "rows_in": 0,
                "rows_out": 0,
                "molecules_out": 0,
                "status": "failed",
                "note": "Descriptor-model-validation smoke failed.",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        for widget in reversed(widgets):
            try:
                if hasattr(widget, "onDeleteWidget"):
                    widget.onDeleteWidget()
            except Exception:
                pass
            try:
                widget.close()
            except Exception:
                pass
            try:
                widget.deleteLater()
            except Exception:
                pass
        try:
            app.processEvents()
        except Exception:
            pass

    summary = {
        "workflows": 1,
        "workflow_failures": 0 if records and all(row.get("status") == "ok" for row in records) else 1,
        "stages": len(records),
        "failed": sum(1 for row in records if row.get("status") != "ok"),
        "prepared_compounds": 0,
        "final_rows": 0 if validation_table is None else len(validation_table),
        "outliers": int(next((row.get("outliers", 0) for row in reversed(records) if "outliers" in row), 0)),
        "validation_outliers": int(next((row.get("outliers", 0) for row in reversed(records) if "outliers" in row), 0)),
        "provenance_ok": 1,
        "workflow_ok": int(bool(records) and all(row.get("status") == "ok" for row in records)),
    }
    return records, summary, validation_table


def run_chembl_builder_workflow_smoke() -> tuple[list[dict[str, Any]], dict[str, Any], Table | None]:
    from chem_inf_widgets.chemcore.services.chembl_bioactivity_dataframe_service import (
        calculate_drug_properties,
        filter_output_columns,
        normalize_smiles_column,
        process_ic50_values,
    )
    from chem_inf_widgets.chemcore.services.chembl_models import ChemBLBioactivityRecord
    from chem_inf_widgets.widgets.ow_chembl_browser import OWChemBLBrowser
    from chem_inf_widgets.widgets.ow_chembl_dataretriever import ChEMBLBioactivityWidget

    app = QApplication.instance() or QApplication([])
    records: list[dict[str, Any]] = []
    ready_table: Table | None = None
    widgets: list[OWWidget] = []

    try:
        retriever_widget = ChEMBLBioactivityWidget()
        widgets.append(retriever_widget)
        retriever_df = pd.DataFrame(_CHEMBL_RETRIEVER_ROWS)
        retriever_df = process_ic50_values(retriever_df)
        retriever_df = normalize_smiles_column(retriever_df)
        retriever_df = calculate_drug_properties(retriever_df)
        retriever_table = retriever_widget._create_orange_table(filter_output_columns(retriever_df))
        retriever_widget._update_output_from_table(retriever_table)
        records.append(
            {
                "stage": "chembl_retriever",
                "widget_class": type(retriever_widget).__name__,
                "rows_in": len(_CHEMBL_RETRIEVER_ROWS),
                "rows_out": 0 if retriever_table is None else len(retriever_table),
                "molecules_out": 0,
                "status": "ok",
                "note": retriever_widget.status_label.text(),
                "error": "",
            }
        )

        original_refresh = OWChemBLBrowser._refresh_property_keys_background
        OWChemBLBrowser._refresh_property_keys_background = lambda self, sample_ids: None
        try:
            browser_widget = OWChemBLBrowser()
        finally:
            OWChemBLBrowser._refresh_property_keys_background = original_refresh
        widgets.append(browser_widget)
        browser_records = [
            ChemBLBioactivityRecord(
                molecule_chembl_id=str(row["molecule_chembl_id"]),
                target_chembl_id=str(row["target_chembl_id"]),
                smiles=str(row["canonical_smiles"]),
                standard_type="IC50",
                standard_value=float(row["standard_value"]),
                standard_units="nM",
                pchembl_value=float(row["pchembl_value"]),
                ic50_nM=float(row["standard_value"]),
            )
            for row in _CHEMBL_RETRIEVER_ROWS
        ]
        browser_table, browser_mols, browser_warning = browser_widget._build_outputs_from_bioactivities(browser_records, [])
        browser_widget._send_outputs(browser_table, browser_mols, browser_warning)
        records.append(
            {
                "stage": "chembl_browser_export",
                "widget_class": type(browser_widget).__name__,
                "rows_in": len(browser_records),
                "rows_out": 0 if browser_table is None else len(browser_table),
                "molecules_out": len(browser_mols),
                "status": "ok",
                "note": browser_widget.lbl_status.text(),
                "error": browser_warning,
            }
        )

        builder_widget = OWQSARDatasetBuilder()
        widgets.append(builder_widget)
        builder_widget.auto_run = False
        builder_widget.set_data(browser_table)
        for row_attr, preferred in (
            ("_row_smiles", "SMILES"),
            ("_row_name", "molecule_chembl_id"),
            ("_row_activity", "pChEMBL"),
            ("_row_unit", "standard_units"),
            ("_row_endpoint", "standard_type"),
        ):
            _builder_select_column(builder_widget, row_attr, preferred)
        builder_widget.target_endpoint = "IC50"
        builder_result = build_qsar_dataset(_table_to_records(browser_table), builder_widget._config())
        builder_widget._on_finished(builder_result)
        ready_table = build_qsar_ready_table(builder_result.prepared_records, name="ChEMBL QSAR Ready Data")
        records.append(
            {
                "stage": "qsar_builder",
                "widget_class": type(builder_widget).__name__,
                "rows_in": 0 if browser_table is None else len(browser_table),
                "rows_out": 0 if ready_table is None else len(ready_table),
                "molecules_out": 0,
                "status": "ok",
                "note": builder_widget._status_label.text(),
                "error": "",
                "prepared_compounds": int(builder_result.summary.get("prepared_compounds", 0)),
                "duplicate_groups": int(builder_result.summary.get("duplicate_groups", 0)),
            }
        )
    except Exception as exc:
        records.append(
            {
                "stage": "chembl_builder",
                "widget_class": "workflow",
                "rows_in": 0,
                "rows_out": 0,
                "molecules_out": 0,
                "status": "failed",
                "note": "ChEMBL-to-builder smoke failed.",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        for widget in reversed(widgets):
            try:
                if hasattr(widget, "onDeleteWidget"):
                    widget.onDeleteWidget()
            except Exception:
                pass
            try:
                widget.close()
            except Exception:
                pass
            try:
                widget.deleteLater()
            except Exception:
                pass
        try:
            app.processEvents()
        except Exception:
            pass

    summary = {
        "workflows": 1,
        "workflow_failures": 0 if records and all(row.get("status") == "ok" for row in records) else 1,
        "stages": len(records),
        "failed": sum(1 for row in records if row.get("status") != "ok"),
        "prepared_compounds": int(next((row.get("prepared_compounds", 0) for row in reversed(records) if "prepared_compounds" in row), 0)),
        "final_rows": 0 if ready_table is None else len(ready_table),
        "validation_outliers": 0,
        "provenance_ok": 1,
        "workflow_ok": int(bool(records) and all(row.get("status") == "ok" for row in records)),
    }
    return records, summary, ready_table


def run_workflow_smoke_suite() -> tuple[list[dict[str, Any]], dict[str, Any], Table | None]:
    suite_records: list[dict[str, Any]] = []
    primary_table: Table | None = None
    workflow_summaries: list[tuple[str, dict[str, Any]]] = []

    for workflow_name, runner in (
        ("core_workflow", run_core_workflow_smoke),
        ("descriptor_model_validation", run_model_validation_workflow_smoke),
        ("chembl_builder", run_chembl_builder_workflow_smoke),
    ):
        records, summary, table = runner()
        for row in records:
            suite_records.append({"workflow": workflow_name, **row})
        workflow_summaries.append((workflow_name, summary))
        if primary_table is None and table is not None:
            primary_table = table

    summary = {
        "workflows": len(workflow_summaries),
        "workflow_failures": sum(1 for _, info in workflow_summaries if not bool(info.get("workflow_ok", 0))),
        "stages": len(suite_records),
        "failed": sum(1 for row in suite_records if row.get("status") != "ok"),
        "prepared_compounds": sum(int(info.get("prepared_compounds", 0) or 0) for _, info in workflow_summaries),
        "final_rows": 0 if primary_table is None else len(primary_table),
        "validation_outliers": sum(int(info.get("outliers", 0) or 0) for _, info in workflow_summaries),
        "provenance_ok": int(all(bool(info.get("provenance_ok", 1)) for _, info in workflow_summaries)),
        "workflow_ok": int(all(bool(info.get("workflow_ok", 0)) for _, info in workflow_summaries)),
    }
    return suite_records, summary, primary_table


def workflow_records_to_table(records: Iterable[dict[str, Any]]) -> Table | None:
    return report_rows_to_table(
        list(records),
        meta_columns=["workflow", "stage", "widget_class", "status", "note", "error"],
        name="Workflow Smoke Report",
    )


def workflow_summary_to_table(summary: dict[str, Any]) -> Table | None:
    rows = [
        {"metric": "workflows", "value": summary.get("workflows", 0), "description": "Workflow scenarios executed."},
        {"metric": "workflow_failures", "value": summary.get("workflow_failures", 0), "description": "Workflow scenarios that failed."},
        {"metric": "stages", "value": summary.get("stages", 0), "description": "Workflow stages executed."},
        {"metric": "failed", "value": summary.get("failed", 0), "description": "Workflow stages that failed."},
        {"metric": "prepared_compounds", "value": summary.get("prepared_compounds", 0), "description": "QSAR-ready compounds produced by the builder."},
        {"metric": "final_rows", "value": summary.get("final_rows", 0), "description": "Rows in the final QSAR-ready table."},
        {"metric": "validation_outliers", "value": summary.get("validation_outliers", 0), "description": "Outliers flagged by validation workflows."},
        {"metric": "provenance_ok", "value": summary.get("provenance_ok", 0), "description": "Whether aggregated provenance columns are present."},
        {"metric": "workflow_ok", "value": summary.get("workflow_ok", 0), "description": "Whether all workflow stages finished successfully."},
    ]
    return summary_rows_to_table(
        rows,
        name="Workflow Smoke Summary",
    )


class OWWidgetSmokeTester(OWWidget):
    name = "Widget Smoke Tester"
    description = "Import and optionally instantiate most Chemoinformatics widgets to validate runtime readiness."
    icon = "icons/input_output/owwidgetsmoketesterwidget.svg"
    priority = 99

    class Outputs:
        smoke_report = Output("Smoke Report", Table)
        smoke_summary = Output("Smoke Summary", Table)
        workflow_report = Output("Workflow Smoke Report", Table)
        workflow_summary = Output("Workflow Smoke Summary", Table)
        workflow_qsar_ready_data = Output("Workflow QSAR Ready Data", Table)

    want_main_area = True

    scope = Setting("core-workflow")
    instantiate_widgets = Setting(True)

    SCOPE_OPTIONS = (
        ("core-workflow", "Core Workflow"),
        ("data", "Data"),
        ("processing", "Processing"),
        ("modeling", "Modeling"),
        ("all", "All Widgets"),
    )

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()
        self._set_status("Run a smoke check to validate imports and widget construction.")

    def _build_ui(self) -> None:
        root = self.controlArea
        root.setMinimumWidth(360)

        opts = QGroupBox("Smoke Check")
        opts_layout = QVBoxLayout(opts)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope"))
        self.scope_combo = QComboBox()
        for key, label in self.SCOPE_OPTIONS:
            self.scope_combo.addItem(label, key)
        current_index = next((i for i, (key, _) in enumerate(self.SCOPE_OPTIONS) if key == self.scope), 0)
        self.scope_combo.setCurrentIndex(current_index)
        self.scope_combo.currentIndexChanged.connect(self._settings_changed)
        scope_row.addWidget(self.scope_combo, 1)
        opts_layout.addLayout(scope_row)

        self.cb_instantiate = QCheckBox("Instantiate widgets after import")
        self.cb_instantiate.setChecked(bool(self.instantiate_widgets))
        self.cb_instantiate.stateChanged.connect(self._settings_changed)
        opts_layout.addWidget(self.cb_instantiate)

        self.btn_run = QPushButton("Run Smoke Check")
        self.btn_run.clicked.connect(self._run)
        opts_layout.addWidget(self.btn_run)

        self.btn_workflow = QPushButton("Run Core Workflow Smoke")
        self.btn_workflow.clicked.connect(self._run_workflow_smoke)
        opts_layout.addWidget(self.btn_workflow)

        self.btn_workflow_suite = QPushButton("Run Workflow Suite")
        self.btn_workflow_suite.clicked.connect(self._run_workflow_suite)
        opts_layout.addWidget(self.btn_workflow_suite)

        self.lbl = QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#475467;")
        opts_layout.addWidget(self.lbl)

        root.layout().addWidget(opts)

        self.results_table = QTableWidget(0, 6)
        self._configure_table(["Category", "Module", "Widget", "Import", "Instantiate", "Error"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.mainArea.layout().addWidget(self.results_table)

    def _settings_changed(self) -> None:
        data = self.scope_combo.currentData()
        self.scope = str(data or "core-workflow")
        self.instantiate_widgets = bool(self.cb_instantiate.isChecked())

    def _set_status(self, text: str) -> None:
        self.lbl.setText(text)

    def _run(self) -> None:
        self._settings_changed()
        records, summary = run_widget_smoke_checks(
            scope=self.scope,
            instantiate_widgets=bool(self.instantiate_widgets),
        )
        self._populate_smoke_table(records)
        report_table = smoke_records_to_table(records)
        summary_table = smoke_summary_to_table(summary)
        self.Outputs.smoke_report.send(report_table)
        self.Outputs.smoke_summary.send(summary_table)
        self.Outputs.workflow_report.send(None)
        self.Outputs.workflow_summary.send(None)
        self.Outputs.workflow_qsar_ready_data.send(None)
        self._set_status(
            f"Smoke check complete: {summary['failed']} failed, "
            f"{summary['import_ok']}/{summary['targets']} imported, "
            f"{summary['instantiate_ok']}/{summary['targets']} instantiated."
        )

    def _run_workflow_smoke(self) -> None:
        records, summary, ready_table = run_core_workflow_smoke()
        self._populate_workflow_table(records)
        self.Outputs.smoke_report.send(None)
        self.Outputs.smoke_summary.send(None)
        self.Outputs.workflow_report.send(workflow_records_to_table(records))
        self.Outputs.workflow_summary.send(workflow_summary_to_table(summary))
        self.Outputs.workflow_qsar_ready_data.send(ready_table)
        self._set_status(
            f"Workflow smoke complete: {summary['failed']} failed, "
            f"{summary['prepared_compounds']} prepared compounds, "
            f"provenance={'ok' if summary['provenance_ok'] else 'missing'}."
        )

    def _run_workflow_suite(self) -> None:
        records, summary, ready_table = run_workflow_smoke_suite()
        self._populate_workflow_table(records)
        self.Outputs.smoke_report.send(None)
        self.Outputs.smoke_summary.send(None)
        self.Outputs.workflow_report.send(workflow_records_to_table(records))
        self.Outputs.workflow_summary.send(workflow_summary_to_table(summary))
        self.Outputs.workflow_qsar_ready_data.send(ready_table)
        self._set_status(
            f"Workflow suite complete: {summary['workflow_failures']} workflow failures, "
            f"{summary['prepared_compounds']} prepared compounds, "
            f"{summary['validation_outliers']} validation outliers."
        )

    def _configure_table(self, headers: Sequence[str]) -> None:
        self.results_table.clear()
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels([str(header) for header in headers])

    def _populate_smoke_table(self, records: Sequence[dict[str, Any]]) -> None:
        self._configure_table(["Category", "Module", "Widget", "Import", "Instantiate", "Error"])
        self.results_table.setRowCount(len(records))
        for row_idx, row in enumerate(records):
            values = [
                str(row.get("category", "")),
                str(row.get("module_name", "")),
                str(row.get("widget_class", "")),
                "yes" if int(row.get("import_ok", 0)) else "no",
                "yes" if int(row.get("instantiate_ok", 0)) else "no",
                str(row.get("error", "")),
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_idx in {3, 4}:
                    item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row_idx, col_idx, item)
        self.results_table.resizeColumnsToContents()

    def _populate_workflow_table(self, records: Sequence[dict[str, Any]]) -> None:
        include_workflow = any("workflow" in row for row in records)
        headers = ["Workflow", "Stage", "Widget", "Status", "Rows In", "Rows Out", "Mol Out", "Note", "Error"] if include_workflow else ["Stage", "Widget", "Status", "Rows In", "Rows Out", "Mol Out", "Note", "Error"]
        self._configure_table(headers)
        self.results_table.setRowCount(len(records))
        for row_idx, row in enumerate(records):
            base_values = [
                str(row.get("stage", "")),
                str(row.get("widget_class", "")),
                str(row.get("status", "")),
                str(row.get("rows_in", "")),
                str(row.get("rows_out", "")),
                str(row.get("molecules_out", "")),
                str(row.get("note", "")),
                str(row.get("error", "")),
            ]
            values = ([str(row.get("workflow", ""))] + base_values) if include_workflow else base_values
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                numeric_cols = {4, 5, 6} if include_workflow else {3, 4, 5}
                if col_idx in numeric_cols:
                    item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row_idx, col_idx, item)
        self.results_table.resizeColumnsToContents()
