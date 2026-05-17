from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.widgets import LassoSelector, RectangleSelector

from chem_inf_widgets.chemcore.services import qsar_regression_service as qsar_service


@dataclass
class DiagnosticSelectionContext:
    canvas: object
    figure: object
    preds: np.ndarray
    y: np.ndarray
    residuals: np.ndarray
    table: object
    overlay_left: object
    overlay_right: object


@dataclass
class DiagnosticSelectorBundle:
    rect_left: object
    rect_right: object
    lasso_left: object
    lasso_right: object


def build_diagnostic_selection_context(
    *,
    canvas,
    figure,
    preds,
    y,
    residuals,
    table,
    overlay_left,
    overlay_right,
) -> DiagnosticSelectionContext:
    return DiagnosticSelectionContext(
        canvas=canvas,
        figure=figure,
        preds=np.asarray(preds, dtype=float),
        y=np.asarray(y, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        table=table,
        overlay_left=overlay_left,
        overlay_right=overlay_right,
    )


def create_diagnostic_selectors(
    *,
    ax_left,
    ax_right,
    on_rect_left,
    on_rect_right,
    on_lasso_left,
    on_lasso_right,
) -> DiagnosticSelectorBundle:
    selector_left = RectangleSelector(
        ax_left,
        on_rect_left,
        useblit=False,
        button=[1],
        minspanx=1e-9,
        minspany=1e-9,
        spancoords="data",
        interactive=False,
    )
    selector_right = RectangleSelector(
        ax_right,
        on_rect_right,
        useblit=False,
        button=[1],
        minspanx=1e-9,
        minspany=1e-9,
        spancoords="data",
        interactive=False,
    )
    lasso_left = LassoSelector(ax_left, on_lasso_left)
    lasso_right = LassoSelector(ax_right, on_lasso_right)
    return DiagnosticSelectorBundle(
        rect_left=selector_left,
        rect_right=selector_right,
        lasso_left=lasso_left,
        lasso_right=lasso_right,
    )


def set_selector_mode(selectors: DiagnosticSelectorBundle, *, use_lasso: bool) -> None:
    selectors.rect_left.set_active(not use_lasso)
    selectors.rect_right.set_active(not use_lasso)
    selectors.lasso_left.set_active(use_lasso)
    selectors.lasso_right.set_active(use_lasso)


def selection_plot_values(context: DiagnosticSelectionContext, *, left_plot: bool) -> tuple[np.ndarray, np.ndarray]:
    return context.preds, context.y if left_plot else context.residuals


def update_selection_overlays(context: DiagnosticSelectionContext, selected_idx) -> None:
    left_offsets, right_offsets = qsar_service.selection_overlay_offsets(
        context.preds,
        context.y,
        context.residuals,
        selected_idx,
    )
    context.overlay_left.set_offsets(left_offsets)
    context.overlay_right.set_offsets(right_offsets)
    context.canvas.draw_idle()


def clear_selection_overlays(context: DiagnosticSelectionContext) -> None:
    empty = np.empty((0, 2))
    context.overlay_left.set_offsets(empty)
    context.overlay_right.set_offsets(empty)
    context.canvas.draw_idle()
