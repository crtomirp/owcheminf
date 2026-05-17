import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from chem_inf_widgets.chemcore.qsar.mlr_selection import ADConfig  # noqa: E402
from chem_inf_widgets.chemcore.services.applicability_domain_service import (  # noqa: E402
    fit_applicability_domain,
    score_applicability_domain,
)


class ApplicabilityDomainServiceTests(unittest.TestCase):
    def setUp(self):
        self.X_ref = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.2, 0.1],
                [0.15, 0.18],
                [0.05, 0.12],
            ],
            dtype=float,
        )
        self.feature_names = ["d1", "d2"]

    def test_fit_and_score_return_expected_shapes(self):
        fit = fit_applicability_domain(self.X_ref, self.feature_names, ad_cfg=ADConfig())
        prediction = score_applicability_domain(fit, self.X_ref)

        self.assertEqual(prediction.leverage.shape, (5,))
        self.assertEqual(prediction.in_ad.shape, (5,))
        self.assertTrue(np.all(prediction.in_ad))

    def test_far_query_point_is_outside_domain(self):
        fit = fit_applicability_domain(
            self.X_ref,
            self.feature_names,
            ad_cfg=ADConfig(use_williams=True, use_knn=True, use_mahalanobis=True, combine_mode="and"),
        )
        X_query = np.array([[0.12, 0.11], [5.0, 5.0]], dtype=float)
        prediction = score_applicability_domain(fit, X_query)

        self.assertTrue(prediction.in_ad[0])
        self.assertFalse(prediction.in_ad[1])
        self.assertGreater(prediction.leverage[1], fit.h_star)

    def test_no_enabled_methods_defaults_to_all_true(self):
        fit = fit_applicability_domain(
            self.X_ref,
            self.feature_names,
            ad_cfg=ADConfig(use_williams=False, use_knn=False, use_mahalanobis=False),
        )
        prediction = score_applicability_domain(fit, np.array([[100.0, 100.0]], dtype=float))

        self.assertTrue(prediction.in_ad[0])
        self.assertTrue(prediction.in_ad_williams[0])

    def test_missing_values_are_imputed(self):
        X_ref = self.X_ref.copy()
        X_ref[0, 1] = np.nan
        fit = fit_applicability_domain(X_ref, self.feature_names, ad_cfg=ADConfig(use_williams=True))
        prediction = score_applicability_domain(fit, np.array([[0.1, np.nan]], dtype=float))

        self.assertEqual(prediction.leverage.shape, (1,))
        self.assertTrue(np.isfinite(prediction.leverage[0]))


if __name__ == "__main__":
    unittest.main()
