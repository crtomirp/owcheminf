import sys
import unittest
from pathlib import Path

import numpy as np
from Orange.data import ContinuousVariable, Domain, StringVariable, Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.services.qsar_regression_service import (  # noqa: E402
    available_algorithms,
    build_diagnostic_plot_spec,
    build_feature_inspection_payload,
    build_selection_gallery_payload,
    build_selection_publish_payload,
    build_report_context,
    build_report_html_from_context,
    build_run_config,
    build_compound_previews,
    build_completed_status_text,
    build_pdf_export_empty_status_text,
    build_pdf_export_error_status_text,
    build_pdf_export_success_status_text,
    build_pdf_report_figure_from_context,
    build_pdf_report_text,
    build_pdf_report_text_from_context,
    build_report_html,
    run_qsar_regression,
    build_waiting_report_html,
    build_waiting_status_text,
    build_cancelled_status_text,
    build_error_status_text,
    collect_pdf_export_figures,
    diagnostic_payloads_from_result,
    find_name_var,
    find_smiles_var,
    prepare_diagnostic_plot_data,
    build_run_config,
)


class QSARRegressionServiceTests(unittest.TestCase):
    class _DummyPipeline:
        def __init__(self, preds):
            self._preds = np.asarray(preds, dtype=float)

        def predict(self, X):
            return self._preds

    @staticmethod
    def _numeric_qsar_table():
        attrs = [
            ContinuousVariable("d1"),
            ContinuousVariable("d2"),
            ContinuousVariable("d3"),
        ]
        target = ContinuousVariable("pActivity")
        domain = Domain(attrs, class_vars=[target], metas=[StringVariable("compound_id")])
        X = np.array(
            [
                [0.1, 1.2, 3.1],
                [0.3, 1.0, 2.8],
                [0.5, 0.9, 2.6],
                [0.7, 0.8, 2.4],
                [0.9, 0.7, 2.2],
                [1.1, 0.5, 2.0],
                [1.3, 0.4, 1.8],
                [1.5, 0.3, 1.6],
                [1.7, 0.2, 1.4],
                [1.9, 0.1, 1.2],
            ],
            dtype=float,
        )
        y = np.array([4.1, 4.4, 4.8, 5.0, 5.3, 5.7, 6.0, 6.3, 6.6, 6.9], dtype=float)
        metas = np.array([[f"M{i:03d}"] for i in range(1, len(y) + 1)], dtype=object)
        return Table.from_numpy(domain, X=X, Y=y.reshape(-1, 1), metas=metas)

    def test_find_smiles_and_name_vars(self):
        smiles_var = StringVariable("SMILES")
        name_var = StringVariable("Name")
        domain = Domain([], metas=[smiles_var, name_var])
        table = Table.from_numpy(
            domain,
            X=np.zeros((1, 0), dtype=float),
            metas=np.array([["CCO", "ethanol"]], dtype=object),
        )

        self.assertEqual(find_smiles_var(table).name, "SMILES")
        self.assertEqual(find_name_var(table).name, "Name")

    def test_build_run_config(self):
        config = build_run_config(
            selected_algorithm=1,
            normalization_method=2,
            imputation_method=3,
            cv_folds=7,
            test_size=0.25,
            tuning_method=1,
            n_iter=20,
            hyperparameters='{"a": 1}',
            enable_feature_selection=True,
            num_features=15,
            algorithms=[("Dummy", object)],
        )

        self.assertEqual(config.selected_algorithm, 1)
        self.assertEqual(config.normalization_method, 2)
        self.assertEqual(config.num_features, 15)
        self.assertTrue(config.enable_feature_selection)

    def test_run_qsar_regression_respects_selected_algorithm_when_auto_is_off(self):
        table = self._numeric_qsar_table()
        algorithms = available_algorithms()

        rf_result = run_qsar_regression(
            table,
            None,
            build_run_config(
                selected_algorithm=0,
                normalization_method=0,
                imputation_method=1,
                cv_folds=2,
                test_size=0.3,
                tuning_method=0,
                n_iter=5,
                hyperparameters="",
                enable_feature_selection=False,
                num_features=3,
                algorithms=algorithms,
                enable_auto_qsar=False,
            ),
        )
        svr_result = run_qsar_regression(
            table,
            None,
            build_run_config(
                selected_algorithm=1,
                normalization_method=0,
                imputation_method=1,
                cv_folds=2,
                test_size=0.3,
                tuning_method=0,
                n_iter=5,
                hyperparameters="",
                enable_feature_selection=False,
                num_features=3,
                algorithms=algorithms,
                enable_auto_qsar=False,
            ),
        )

        self.assertEqual(type(rf_result["pipeline"].named_steps["regressor"]).__name__, "RandomForestRegressor")
        self.assertEqual(type(svr_result["pipeline"].named_steps["regressor"]).__name__, "SVR")
        self.assertEqual(rf_result["model_name"], "Random Forest")
        self.assertEqual(svr_result["model_name"], "Support Vector Regression")
        self.assertFalse(np.allclose(rf_result["test_table"].X[:, -1], svr_result["test_table"].X[:, -1]))

    def test_run_qsar_regression_pls_skips_ols_style_linear_diagnostics(self):
        table = self._numeric_qsar_table()
        algorithms = available_algorithms()
        pls_idx = next(i for i, (name, _) in enumerate(algorithms) if name == "PLS Regression")

        pls_result = run_qsar_regression(
            table,
            None,
            build_run_config(
                selected_algorithm=pls_idx,
                normalization_method=0,
                imputation_method=1,
                cv_folds=2,
                test_size=0.3,
                tuning_method=0,
                n_iter=5,
                hyperparameters="",
                enable_feature_selection=False,
                num_features=3,
                algorithms=algorithms,
                enable_auto_qsar=False,
            ),
        )

        self.assertEqual(pls_result["model_name"], "PLS Regression")
        self.assertNotIn("coef_stats", pls_result)
        self.assertNotIn("vifs", pls_result)
        self.assertIn("PLS latent-space model", pls_result.get("linear_diagnostics_note", ""))

    def test_run_qsar_regression_regularized_models_skip_ols_style_linear_diagnostics(self):
        table = self._numeric_qsar_table()
        algorithms = available_algorithms()

        for model_name in ("Lasso Regression", "Ridge Regression", "Elastic Net"):
            idx = next(i for i, (name, _) in enumerate(algorithms) if name == model_name)
            result = run_qsar_regression(
                table,
                None,
                build_run_config(
                    selected_algorithm=idx,
                    normalization_method=0,
                    imputation_method=1,
                    cv_folds=2,
                    test_size=0.3,
                    tuning_method=0,
                    n_iter=5,
                    hyperparameters="",
                    enable_feature_selection=False,
                    num_features=3,
                    algorithms=algorithms,
                    enable_auto_qsar=False,
                ),
            )

            self.assertEqual(result["model_name"], model_name)
            self.assertNotIn("coef_stats", result)
            self.assertNotIn("vifs", result)
            self.assertIn("regularized model", result.get("linear_diagnostics_note", ""))

    def test_build_compound_previews_skips_invalid_smiles(self):
        smiles_var = StringVariable("SMILES")
        name_var = StringVariable("Name")
        domain = Domain([], metas=[smiles_var, name_var])
        table = Table.from_numpy(
            domain,
            X=np.zeros((2, 0), dtype=float),
            metas=np.array([["CCO", "ethanol"], ["not_a_smiles", "bad"]], dtype=object),
        )

        previews = build_compound_previews(table, max_preview=12)

        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0].title, "ethanol")
        self.assertTrue(previews[0].png_bytes)

    def test_build_report_html_contains_model_summary(self):
        html = build_report_html(
            model_name="Random Forest",
            total_descriptors=100,
            descriptors_used=12,
            cv_score=0.82,
            train_metrics={"R²": 0.9, "RMSE": 0.5, "MAE": 0.4, "Median AE": 0.3, "Explained Variance": 0.91},
            test_metrics={"R²": 0.8, "RMSE": 0.7, "MAE": 0.6, "Median AE": 0.5, "Explained Variance": 0.81},
            external_metrics={},
        )

        self.assertIn("Random Forest", html)
        self.assertIn("Descriptors Used", html)
        self.assertIn("0.820", html)

    def test_status_helpers_and_waiting_html(self):
        self.assertIn("Random Forest", build_waiting_status_text("Random Forest"))
        self.assertIn("in progress", build_waiting_report_html())
        self.assertEqual(build_cancelled_status_text(), "Calculation cancelled.")
        self.assertIn("completed", build_completed_status_text("Random Forest", "metrics"))
        self.assertEqual(build_error_status_text("boom"), "Error: boom")
        self.assertEqual(build_pdf_export_success_status_text(), "PDF Exported Successfully.")
        self.assertEqual(build_pdf_export_empty_status_text(), "No QSAR results available to export.")
        self.assertEqual(build_pdf_export_error_status_text("boom"), "Error exporting PDF: boom")

    def test_build_pdf_report_text_contains_metrics(self):
        report_text = build_pdf_report_text(
            model_name="Random Forest",
            total_descriptors=100,
            descriptors_used=12,
            cv_score=0.82,
            train_metrics={"R²": 0.9, "RMSE": 0.5},
            test_metrics={"R²": 0.8},
            external_metrics={"MAE": 0.7},
        )

        self.assertIn("Model: Random Forest", report_text)
        self.assertIn("Total Descriptors: 100", report_text)
        self.assertIn("CV R²: 0.820", report_text)
        self.assertIn("Training Metrics:", report_text)
        self.assertIn("R²: 0.900", report_text)
        self.assertIn("External Metrics:", report_text)
        self.assertIn("MAE: 0.700", report_text)

    def test_report_context_helpers(self):
        context = build_report_context(
            model_name="Random Forest",
            total_descriptors=100,
            descriptors_used=12,
            cv_score=0.82,
            train_metrics={"R²": 0.9},
            test_metrics={"R²": 0.8},
            external_metrics={"MAE": 0.7},
        )

        html = build_report_html_from_context(context)
        pdf_text = build_pdf_report_text_from_context(context)

        self.assertIn("Random Forest", html)
        self.assertIn("Descriptors Used", html)
        self.assertIn("Model: Random Forest", pdf_text)
        self.assertIn("MAE: 0.700", pdf_text)

        fig = build_pdf_report_figure_from_context(context)
        self.assertEqual(len(fig.axes), 1)

    def test_collect_pdf_export_figures_filters_none(self):
        figures = collect_pdf_export_figures(None, "a", None, "b")
        self.assertEqual(figures, ("a", "b"))

    def test_build_selection_gallery_payload_empty(self):
        payload = build_selection_gallery_payload(None, "train")
        self.assertEqual(payload.more_count, 0)
        self.assertEqual(payload.previews, ())
        self.assertIn("No compounds selected", payload.placeholder_text)

    def test_build_selection_gallery_payload_with_previews(self):
        smiles_var = StringVariable("SMILES")
        name_var = StringVariable("Name")
        domain = Domain([], metas=[smiles_var, name_var])
        table = Table.from_numpy(
            domain,
            X=np.zeros((3, 0), dtype=float),
            metas=np.array(
                [["CCO", "ethanol"], ["c1ccncc1", "pyridine"], ["CCN", "ethylamine"]],
                dtype=object,
            ),
        )

        payload = build_selection_gallery_payload(table, "test", max_preview=2)

        self.assertIsNone(payload.placeholder_text)
        self.assertEqual(len(payload.previews), 2)
        self.assertEqual(payload.more_count, 1)

    def test_build_selection_publish_payload(self):
        smiles_var = StringVariable("SMILES")
        domain = Domain([], metas=[smiles_var])
        table = Table.from_numpy(
            domain,
            X=np.zeros((3, 0), dtype=float),
            metas=np.array([["CCO"], ["c1ccncc1"], ["CCN"]], dtype=object),
        )

        payload = build_selection_publish_payload(
            model_name="Random Forest",
            dataset_type="train",
            table=table,
            selected_idx=np.array([0, 2], dtype=int),
            max_preview=5,
        )

        self.assertEqual(len(payload.selected_table), 2)
        self.assertEqual(len(payload.gallery.previews), 2)
        self.assertEqual(payload.gallery.more_count, 0)
        self.assertIn("Selected 2 compounds from train diagnostics", payload.status_text)

    def test_prepare_diagnostic_plot_data_regression(self):
        pipeline = self._DummyPipeline([1.0, 2.0, 10.0])
        X = np.zeros((3, 2), dtype=float)
        y = np.array([1.1, 1.9, 0.0], dtype=float)

        diagnostic = prepare_diagnostic_plot_data(X, y, pipeline, is_classification=False)

        self.assertEqual(diagnostic.preds.tolist(), [1.0, 2.0, 10.0])
        self.assertEqual(diagnostic.actuals.tolist(), [1.1, 1.9, 0.0])
        self.assertEqual(diagnostic.residuals.shape[0], 3)
        self.assertTrue(bool(diagnostic.outlier_mask[-1]))
        self.assertFalse(diagnostic.is_classification)

        plot_spec = build_diagnostic_plot_spec(diagnostic)
        self.assertEqual(plot_spec.left_title, "Predicted vs Actual")
        self.assertEqual(len(plot_spec.left_series), 2)
        self.assertEqual(plot_spec.left_series[0].label, "Inliers")
        self.assertEqual(plot_spec.left_series[1].label, "Outliers")
        self.assertTrue(plot_spec.show_legends)

    def test_prepare_diagnostic_plot_data_classification(self):
        pipeline = self._DummyPipeline([0.0, 1.0, 1.0])
        X = np.zeros((3, 2), dtype=float)
        y = np.array([0.0, 0.0, 1.0], dtype=float)

        diagnostic = prepare_diagnostic_plot_data(X, y, pipeline, is_classification=True)

        self.assertEqual(diagnostic.residuals.tolist(), [0.0, 1.0, 0.0])
        self.assertTrue(np.all(diagnostic.inlier_mask))
        self.assertFalse(np.any(diagnostic.outlier_mask))
        self.assertTrue(diagnostic.is_classification)

        plot_spec = build_diagnostic_plot_spec(diagnostic)
        self.assertEqual(plot_spec.right_title, "Misclassifications (1 if error)")
        self.assertEqual(len(plot_spec.left_series), 1)
        self.assertEqual(plot_spec.left_series[0].label, "Observations")
        self.assertFalse(plot_spec.show_legends)

    def test_diagnostic_payloads_from_result(self):
        result = {
            "X_train": np.zeros((2, 1)),
            "y_train": np.array([1.0, 2.0]),
            "X_test": np.zeros((1, 1)),
            "y_test": np.array([3.0]),
            "pipeline": object(),
            "is_classification": False,
            "train_table": "train-table",
            "test_table": "test-table",
            "external_table": "ext-table",
            "X_ext": np.zeros((1, 1)),
            "y_ext": np.array([4.0]),
        }

        payloads = diagnostic_payloads_from_result(result, include_external=True)

        self.assertEqual([p.dataset_type for p in payloads], ["train", "test", "external"])
        self.assertEqual(payloads[0].result_table, "train-table")
        self.assertEqual(payloads[2].result_table, "ext-table")

    def test_build_feature_inspection_payload_without_values(self):
        class _NoValueEstimator:
            pass

        class _DummyPipeline:
            named_steps = {"regressor": _NoValueEstimator()}

            def __getitem__(self, index):
                return self.named_steps["regressor"]

        payload = build_feature_inspection_payload(
            {
                "pipeline": _DummyPipeline(),
                "feature_names": ["d1", "d2"],
            },
            model_name="Dummy Model",
        )

        self.assertTrue(payload.available)
        self.assertIsNone(payload.values)
        self.assertIn("Dummy Model", payload.message_html)
        self.assertEqual(payload.tab_title, "Features (2)")

    def test_build_feature_inspection_payload_with_coefficients_and_stats(self):
        class _Estimator:
            coef_ = np.array([0.2, -1.5, 0.7], dtype=float)

        class _DummyPipeline:
            named_steps = {"regressor": _Estimator()}

            def __getitem__(self, index):
                return self.named_steps["regressor"]

        payload = build_feature_inspection_payload(
            {
                "pipeline": _DummyPipeline(),
                "feature_names": ["a", "b", "c"],
                "coef_stats": {
                    "beta": np.array([0.0, 0.2, -1.5, 0.7], dtype=float),
                    "se": np.array([0.1, 0.01, 0.02, 0.03], dtype=float),
                    "t": np.array([0.0, 2.0, -3.0, 4.0], dtype=float),
                    "p": np.array([1.0, 0.05, 0.01, 0.001], dtype=float),
                },
                "vifs": np.array([1.2, 5.5, 2.1], dtype=float),
            },
            model_name="Linear Model",
        )

        self.assertTrue(payload.available)
        self.assertEqual(payload.value_label, "Coefficient")
        self.assertEqual(list(payload.names), ["b", "c", "a"])
        self.assertAlmostEqual(float(payload.values[0]), -1.5, places=6)
        self.assertEqual(payload.chart_names[0], "b")
        self.assertEqual(payload.tab_title, "Features (3)")


if __name__ == "__main__":
    unittest.main()
