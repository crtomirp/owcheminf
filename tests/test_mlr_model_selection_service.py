import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chem_inf_widgets.chemcore.services import mlr_model_selection_service as mlr_service  # noqa: E402

try:
    from Orange.data import ContinuousVariable, Domain, Table
except Exception:  # pragma: no cover
    ContinuousVariable = Domain = Table = None  # type: ignore


@unittest.skipIf(Table is None, "Orange is required for MLR service tests")
class MLRModelSelectionServiceTests(unittest.TestCase):
    def test_results_table_adds_prediction_columns(self):
        domain = Domain([ContinuousVariable("x1"), ContinuousVariable("x2")])
        table = Table.from_numpy(domain, X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
        out = mlr_service.results_table(
            table,
            np.array([1.5, 2.5], dtype=float),
            np.array([1.4, 2.6], dtype=float),
            np.array([0.1, 0.2], dtype=float),
            np.array([0.5, -0.5], dtype=float),
            np.array([True, False]),
            prefix="train",
        )
        names = [var.name for var in out.domain.attributes]
        self.assertIn("train_y", names)
        self.assertIn("train_y_pred", names)
        self.assertIn("train_in_AD", names)

    def test_build_summary_html_mentions_selected_descriptors(self):
        html = mlr_service.build_summary_html(
            y_var="pIC50",
            n_train=20,
            n_test=5,
            names_before=120,
            names_after_pre=40,
            selected=["desc_a", "desc_b"],
            train_metrics={"r2": 0.9, "rmse": 0.4, "mae": 0.3},
            test_metrics={"r2": 0.8, "rmse": 0.5, "mae": 0.4},
            ext_metrics=None,
            cv_metrics={"q2": 0.77, "rmse_cv": 0.55},
            h_star=0.5,
            ad_cfg=type(
                "ADCfg",
                (),
                {
                    "use_williams": True,
                    "use_knn": False,
                    "use_mahalanobis": False,
                    "combine_mode": "and",
                    "knn_k": 5,
                    "knn_quantile": 0.95,
                    "maha_alpha": 0.95,
                    "maha_use_chi2": True,
                },
            )(),
            knn_threshold=None,
            maha_threshold=None,
            coef_stats={
                "beta": np.array([0.1, 0.2, -0.3]),
                "se": np.array([0.01, 0.02, 0.03]),
                "t": np.array([10.0, 5.0, -4.0]),
                "p": np.array([0.001, 0.01, 0.02]),
            },
            vifs=np.array([1.1, 2.2]),
            perm_info=None,
            method="forward",
            criterion="cv_r2",
            cv_folds=5,
        )
        self.assertIn("MLR Model Selection", html)
        self.assertIn("desc_a", html)
        self.assertIn("Train", html)


if __name__ == "__main__":
    unittest.main()
