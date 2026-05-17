from __future__ import annotations

import pytest

from chem_inf_widgets.chemcore.services.qsar_diagnostics_contract import (
    SELECTION_TOOL_OPTIONS,
    display_model_name,
    residual_reference_levels,
)


def test_qsar_diagnostics_contract_formats_model_name():
    assert display_model_name("random_forest") == "Random Forest"
    assert display_model_name("ridge") == "Ridge"
    assert display_model_name("") == ""


def test_qsar_diagnostics_contract_exposes_standard_selection_tools():
    assert SELECTION_TOOL_OPTIONS == ("Rectangle", "Lasso")


def test_qsar_diagnostics_contract_reference_levels_include_std_bands():
    levels = residual_reference_levels([0.0, 1.0, 2.0, 3.0])

    assert levels["mean"] == pytest.approx(1.5)
    assert levels["std"] == pytest.approx(1.1180339887)
    assert levels["plus_1std"] == pytest.approx(levels["mean"] + levels["std"])
    assert levels["minus_2std"] == pytest.approx(levels["mean"] - 2.0 * levels["std"])
