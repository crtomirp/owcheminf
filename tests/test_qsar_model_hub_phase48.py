from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table

pytest.importorskip("Orange")
pytest.importorskip("AnyQt")

from AnyQt.QtWidgets import QApplication

from chem_inf_widgets.chemcore.services.qsar_model_hub_service import (
    QSARModelHubConfig,
    train_qsar_model_hub,
)
from chem_inf_widgets.chemcore.services.qsar_prediction_packager_service import (
    QSARPredictionModelBundle,
    write_model_bundle_package,
)
from chem_inf_widgets.widgets.ow_qsar_model_hub import (
    OWQSARModelHub,
    _dataframe_to_orange,
    _input_table_diagnostic,
    _preferred_target_name,
    _residual_reference_levels,
)


_APP = QApplication.instance() or QApplication([])


def _demo_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound_id": [f"C{i:03d}" for i in range(12)],
            "name": [f"cmpd_{i:03d}" for i in range(12)],
            "canonical_smiles": ["CCO", "CCCO", "CCCCO", "CCN", "CCCN", "c1ccccc1O", "c1ccccc1N", "CCOC", "CCS", "CCCl", "CCBr", "CCI"],
            "pActivity": [5.1, 5.4, 5.6, 5.9, 6.0, 6.2, 6.4, 6.7, 6.9, 7.0, 7.2, 7.3],
            "MW": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
            "LogP": [1.0, 1.1, 1.3, 1.4, 1.8, 2.0, 2.1, 2.3, 2.5, 2.8, 3.0, 3.2],
            "TPSA": [10, 12, 15, 16, 19, 20, 22, 23, 25, 26, 29, 31],
        }
    )


def test_qsar_model_hub_predictions_preserve_smiles_and_name():
    result = train_qsar_model_hub(
        _demo_df(),
        QSARModelHubConfig(target_column="pActivity", id_column="compound_id", model_key="random_forest", cv_folds=3),
    )

    assert "canonical_smiles" in result.predictions.columns
    assert "name" in result.predictions.columns
    assert result.predictions["canonical_smiles"].astype(str).str.len().gt(0).all()


def test_qsar_model_hub_prefers_class_var_as_dependent_variable():
    domain = Domain(
        [ContinuousVariable("MW"), ContinuousVariable("LogP"), ContinuousVariable("pActivity")],
        class_vars=[ContinuousVariable("BoilingPoint")],
    )
    table = Table.from_numpy(
        domain,
        X=np.array([[100.0, 1.1, 5.2], [120.0, 1.4, 5.6]], dtype=float),
        Y=np.array([[78.0], [92.0]], dtype=float),
    )

    assert _preferred_target_name(table, OWQSARModelHub._TARGET_CANDIDATES) == "BoilingPoint"


def test_qsar_model_hub_excludes_source_prefixed_provenance_columns_from_features():
    df = _demo_df().copy()
    df["source_MW"] = df["MW"]
    df["source_LogP"] = df["LogP"]
    df["source_qc_flags_all"] = np.arange(len(df), dtype=float)

    result = train_qsar_model_hub(
        df,
        QSARModelHubConfig(
            target_column="pActivity",
            id_column="compound_id",
            model_key="random_forest",
            cv_folds=3,
        ),
    )

    assert "MW" in result.feature_names
    assert "LogP" in result.feature_names
    assert all(not str(name).startswith("source_") for name in result.feature_names)


def test_qsar_model_hub_reference_levels_include_std_bands():
    levels = _residual_reference_levels([0.0, 1.0, 2.0, 3.0])

    assert levels["mean"] == pytest.approx(1.5)
    assert levels["std"] == pytest.approx(1.1180339887)
    assert levels["plus_1std"] == pytest.approx(levels["mean"] + levels["std"])
    assert levels["minus_2std"] == pytest.approx(levels["mean"] - 2.0 * levels["std"])


def test_qsar_model_hub_widget_defaults_to_random_forest_and_rectangle_selection():
    widget = OWQSARModelHub()
    try:
        assert widget._cmb_algo.currentText() == "random_forest"
        assert widget._cmb_selection_tool.currentText() == "Rectangle"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_model_hub_widget_uses_class_var_name_in_target_field():
    widget = OWQSARModelHub()
    try:
        domain = Domain(
            [ContinuousVariable("MW"), ContinuousVariable("LogP"), ContinuousVariable("pActivity")],
            class_vars=[ContinuousVariable("BoilingPoint")],
        )
        table = Table.from_numpy(
            domain,
            X=np.array([[100.0, 1.1, 5.2], [120.0, 1.4, 5.6]], dtype=float),
            Y=np.array([[78.0], [92.0]], dtype=float),
        )
        widget.set_data(table)

        assert widget._ed_target.text() == "BoilingPoint"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_model_hub_does_not_prefer_non_numeric_class_var_as_target():
    widget = OWQSARModelHub()
    try:
        domain = Domain(
            [ContinuousVariable("MW"), ContinuousVariable("LogP"), ContinuousVariable("pActivity")],
            class_vars=[DiscreteVariable("Series", values=("A", "B"))],
        )
        table = Table.from_numpy(
            domain,
            X=np.array([[100.0, 1.1, 5.2], [120.0, 1.4, 5.6]], dtype=float),
            Y=np.array([[0.0], [1.0]], dtype=float),
        )
        widget.set_data(table)

        assert widget._ed_target.text() == "pActivity"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_model_hub_detects_summary_like_builder_output():
    domain = Domain(
        [ContinuousVariable("value")],
        metas=[StringVariable("metric"), StringVariable("description")],
    )
    table = Table.from_numpy(
        domain,
        X=np.array([[12.0], [3.0]], dtype=float),
        metas=np.array(
            [
                ["prepared_compounds", "Prepared compounds"],
                ["rejected_records", "Rejected records"],
            ],
            dtype=object,
        ),
    )

    assert _preferred_target_name(table, OWQSARModelHub._TARGET_CANDIDATES) == ""
    assert "QSAR Ready Data" in _input_table_diagnostic(table)


def test_qsar_model_hub_shows_clear_message_for_summary_like_input():
    widget = OWQSARModelHub()
    try:
        widget.auto_run = False
        domain = Domain(
            [ContinuousVariable("value")],
            metas=[StringVariable("metric"), StringVariable("description")],
        )
        table = Table.from_numpy(
            domain,
            X=np.array([[12.0], [3.0]], dtype=float),
            metas=np.array(
                [
                    ["prepared_compounds", "Prepared compounds"],
                    ["rejected_records", "Rejected records"],
                ],
                dtype=object,
            ),
        )
        widget.set_data(table)
        widget.commit()

        assert widget._lbl_status.text() == "Error"
        assert "QSAR Ready Data" in widget._txt_summary.toPlainText()
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_model_hub_widget_auto_selection_enables_hpo():
    widget = OWQSARModelHub()
    try:
        if "auto" not in widget._MODEL_KEYS:
            pytest.skip("auto/HPO mode not available in this environment")
        widget._cmb_algo.setCurrentIndex(widget._MODEL_KEYS.index("auto"))
        widget._on_settings_changed()

        assert widget._chk_hpo.isChecked() is True
        assert widget._cmb_algo.currentText() == "auto"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_model_hub_widget_disabling_hpo_restores_random_forest():
    widget = OWQSARModelHub()
    try:
        if "auto" not in widget._MODEL_KEYS:
            pytest.skip("auto/HPO mode not available in this environment")
        widget._chk_hpo.setChecked(True)
        widget._on_hpo_toggled(True)
        assert widget._cmb_algo.currentText() == "auto"

        widget._chk_hpo.setChecked(False)
        widget._on_hpo_toggled(False)

        assert widget._cmb_algo.currentText() == "random_forest"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_model_hub_widget_updates_selected_table_tab():
    widget = OWQSARModelHub()
    try:
        table = _dataframe_to_orange(
            pd.DataFrame(
                {
                    "compound_id": ["A", "B"],
                    "split": ["test", "test"],
                    "observed": [5.0, 6.0],
                    "predicted": [5.2, 5.8],
                    "residual": [-0.2, 0.2],
                }
            )
        )
        widget._update_selected_table(table)

        assert widget._tbl_selected.rowCount() == 2
        assert widget._tabs.tabText(widget._tabs.indexOf(widget._selected_tab)) == "Selected (2)"
    finally:
        widget.onDeleteWidget()
        widget.close()


def test_qsar_model_hub_builds_prediction_bundle_with_training_summary(tmp_path):
    result = train_qsar_model_hub(
        _demo_df(),
        QSARModelHubConfig(target_column="pActivity", id_column="compound_id", model_key="ridge", cv_folds=3),
    )
    widget = OWQSARModelHub()
    try:
        widget.target_unit = "log units"
        widget._last_model_name = "Ridge"
        bundle = widget._build_prediction_bundle(result)
        paths = write_model_bundle_package(bundle, tmp_path / "ridge_bundle")

        assert isinstance(bundle, QSARPredictionModelBundle)
        assert bundle.target_label == "pActivity"
        assert bundle.training_rows == result.n_rows_used
        assert bundle.training_summary["target_unit"] == "log units"
        assert bundle.training_summary["model_display_name"] == "Ridge"
        assert bundle.training_summary["selected_feature_names"]
        assert (tmp_path / "ridge_bundle.model.pkl").exists()
        assert Path(paths["manifest_json"]).exists()
    finally:
        widget.onDeleteWidget()
        widget.close()
