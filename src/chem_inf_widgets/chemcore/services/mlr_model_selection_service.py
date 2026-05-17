from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    out = f"{name}_{i}"
    used.add(out)
    return out


def continuous_candidates(data: Table) -> List[ContinuousVariable]:
    out: List[ContinuousVariable] = []
    for variable in list(data.domain.class_vars) + list(data.domain.attributes):
        if getattr(variable, "is_continuous", False):
            out.append(variable)

    seen = set()
    uniq = []
    for variable in out:
        if variable.name not in seen:
            uniq.append(variable)
            seen.add(variable.name)
    return uniq


def extract_xy(
    data: Table,
    y_var: ContinuousVariable,
) -> Tuple[np.ndarray, np.ndarray, List[ContinuousVariable]]:
    x_vars: List[ContinuousVariable] = [
        variable
        for variable in data.domain.attributes
        if getattr(variable, "is_continuous", False) and variable.name != y_var.name
    ]
    if not x_vars:
        raise ValueError("No continuous descriptor columns found in attributes.")

    x_cols = [data.get_column(variable).astype(float) for variable in x_vars]
    X = np.column_stack(x_cols).astype(float)
    y = data.get_column(y_var).astype(float)
    return X, y, x_vars


def extract_x_only(data: Table, x_names: Sequence[str]) -> np.ndarray:
    cols = []
    for name in x_names:
        variable = next((var for var in data.domain.attributes if var.name == name), None)
        if variable is None:
            raise ValueError(f"Test data missing descriptor column: {name}")
        cols.append(data.get_column(variable).astype(float))
    return np.column_stack(cols).astype(float)


def coefficients_table(
    selected: Sequence[str],
    coef_stats: Dict[str, np.ndarray],
    vifs: np.ndarray,
) -> Table:
    terms = ["Intercept"] + list(selected)
    beta = coef_stats["beta"]
    se = coef_stats["se"]
    t = coef_stats["t"]
    p = coef_stats["p"]

    var_beta = ContinuousVariable("beta")
    var_se = ContinuousVariable("se")
    var_t = ContinuousVariable("t")
    var_p = ContinuousVariable("p")
    var_vif = ContinuousVariable("vif")
    term_meta = StringVariable("term")

    X = np.zeros((len(terms), 5), dtype=float)
    X[:, 0] = beta
    X[:, 1] = se
    X[:, 2] = t
    X[:, 3] = p
    X[:, 4] = np.array([np.nan] + list(vifs), dtype=float) if len(terms) > 1 else np.array([np.nan])

    metas = np.array(terms, dtype=object).reshape(-1, 1)
    domain = Domain([var_beta, var_se, var_t, var_p, var_vif], class_vars=None, metas=[term_meta])
    return Table.from_numpy(domain, X, metas=metas)


def fig_to_datauri_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_report_html(summary_html: str, fig_pred, fig_williams, fig_perm) -> str:
    img_pred = fig_to_datauri_png(fig_pred)
    img_williams = fig_to_datauri_png(fig_williams)
    img_perm = fig_to_datauri_png(fig_perm)

    html = []
    html.append("<html><head><meta charset='utf-8'></head><body>")
    html.append(summary_html)
    html.append("<h3>Plots</h3>")
    html.append("<h4>Predicted vs Real</h4>")
    html.append(f"<img src='{img_pred}' style='max-width: 100%; height: auto;'/>")
    html.append("<h4>Williams plot</h4>")
    html.append(f"<img src='{img_williams}' style='max-width: 100%; height: auto;'/>")
    html.append("<h4>Permutation test</h4>")
    html.append(f"<img src='{img_perm}' style='max-width: 100%; height: auto;'/>")
    html.append("</body></html>")
    return "\n".join(html)


def results_table(
    data: Table,
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    leverage: np.ndarray,
    std_resid: Optional[np.ndarray],
    in_ad: np.ndarray,
    prefix: str,
    in_ad_williams: Optional[np.ndarray] = None,
    knn_dist: Optional[np.ndarray] = None,
    in_ad_knn: Optional[np.ndarray] = None,
    maha_d2: Optional[np.ndarray] = None,
    in_ad_maha: Optional[np.ndarray] = None,
) -> Table:
    """Create an Orange Table with predictions and diagnostics, keeping original metas."""
    used = {
        variable.name
        for variable in data.domain.attributes
    } | {
        variable.name
        for variable in data.domain.metas
    } | {
        variable.name
        for variable in data.domain.class_vars
    }

    attrs = []
    if y_true is not None:
        attrs.append(ContinuousVariable(_unique_name(f"{prefix}_y", used)))
    attrs.append(ContinuousVariable(_unique_name(f"{prefix}_y_pred", used)))
    if y_true is not None:
        attrs.append(ContinuousVariable(_unique_name(f"{prefix}_residual", used)))
    attrs.append(ContinuousVariable(_unique_name(f"{prefix}_leverage", used)))
    if std_resid is not None:
        attrs.append(ContinuousVariable(_unique_name(f"{prefix}_std_resid", used)))

    if in_ad_williams is not None:
        attrs.append(DiscreteVariable(_unique_name(f"{prefix}_in_AD_williams", used), values=("False", "True")))

    if knn_dist is not None:
        attrs.append(ContinuousVariable(_unique_name(f"{prefix}_knn_dist", used)))
        attrs.append(DiscreteVariable(_unique_name(f"{prefix}_in_AD_knn", used), values=("False", "True")))

    if maha_d2 is not None:
        attrs.append(ContinuousVariable(_unique_name(f"{prefix}_maha_d2", used)))
        attrs.append(DiscreteVariable(_unique_name(f"{prefix}_in_AD_maha", used), values=("False", "True")))

    attrs.append(DiscreteVariable(_unique_name(f"{prefix}_in_AD", used), values=("False", "True")))

    domain = Domain(attrs, data.domain.class_vars, data.domain.metas)
    out = data.transform(domain)

    cols = []
    if y_true is not None:
        cols.append(y_true.reshape(-1, 1))
        residual = (y_true - y_pred).reshape(-1, 1)
    else:
        residual = None

    cols.append(y_pred.reshape(-1, 1))
    if residual is not None:
        cols.append(residual)
    cols.append(leverage.reshape(-1, 1))
    if std_resid is not None:
        cols.append(std_resid.reshape(-1, 1))

    if in_ad_williams is not None:
        cols.append(in_ad_williams.astype(int).reshape(-1, 1))

    if knn_dist is not None:
        cols.append(knn_dist.reshape(-1, 1))
        cols.append((np.ones(len(knn_dist), dtype=bool) if in_ad_knn is None else in_ad_knn).astype(int).reshape(-1, 1))

    if maha_d2 is not None:
        cols.append(maha_d2.reshape(-1, 1))
        cols.append((np.ones(len(maha_d2), dtype=bool) if in_ad_maha is None else in_ad_maha).astype(int).reshape(-1, 1))

    cols.append(in_ad.astype(int).reshape(-1, 1))

    out.X = np.hstack(cols).astype(float)
    return out


def build_summary_html(
    *,
    y_var: str,
    n_train: int,
    n_test: int,
    names_before: int,
    names_after_pre: int,
    selected: Sequence[str],
    train_metrics: Dict[str, float],
    test_metrics: Optional[Dict[str, float]],
    ext_metrics: Optional[Dict[str, float]],
    cv_metrics: Dict[str, float],
    h_star: float,
    ad_cfg: Any,
    knn_threshold: Optional[float],
    maha_threshold: Optional[float],
    coef_stats: Dict[str, np.ndarray],
    vifs: np.ndarray,
    perm_info: Optional[Dict[str, Any]],
    method: str,
    criterion: str,
    cv_folds: int,
) -> str:
    beta = coef_stats["beta"]
    se = coef_stats["se"]
    t = coef_stats["t"]
    p = coef_stats["p"]

    rows = []
    rows.append("<h2>MLR Model Selection</h2>")
    rows.append(f"<b>Target:</b> {y_var}<br>")
    rows.append(f"<b>Train size:</b> {n_train} &nbsp;&nbsp; <b>Test size:</b> {n_test}<br>")
    rows.append(f"<b>Descriptors:</b> {names_before} → after filters {names_after_pre} → selected {len(selected)}<br>")
    rows.append(f"<b>Selection:</b> method={method}, criterion={criterion}, CV={cv_folds}<br>")

    ad_parts = []
    if ad_cfg.use_williams:
        ad_parts.append(f"Williams: h*={h_star:.4g}")
    if ad_cfg.use_knn and knn_threshold is not None:
        ad_parts.append(f"kNN(k={ad_cfg.knn_k}) thr={knn_threshold:.4g} (q={ad_cfg.knn_quantile:.2f})")
    if ad_cfg.use_mahalanobis and maha_threshold is not None:
        ad_parts.append(f"Mahalanobis thr={maha_threshold:.4g} (α={ad_cfg.maha_alpha:.2f}{', χ²' if ad_cfg.maha_use_chi2 else ''})")
    ad_line = "; ".join(ad_parts) if ad_parts else "(disabled)"
    rows.append(f"<b>Applicability domain:</b> {ad_line}; combine={ad_cfg.combine_mode}; |std resid| ≤ 3 (if Y)<br>")

    rows.append("<h3>Performance</h3>")
    rows.append("<ul>")
    rows.append(f"<li><b>Train</b>: R²={train_metrics['r2']:.4f}, RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}</li>")
    rows.append(f"<li><b>CV</b>: Q²={cv_metrics['q2']:.4f}, RMSEcv={cv_metrics['rmse_cv']:.4f}</li>")
    if test_metrics is not None:
        rows.append(f"<li><b>Test</b>: R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}</li>")
    rows.append("</ul>")

    if ext_metrics is not None:
        rows.append("<h3>External validation (test)</h3>")
        rows.append("<ul>")
        rows.append(f"<li><b>CCC</b>={ext_metrics.get('ccc', float('nan')):.4f}</li>")
        rows.append(
            f"<li><b>Q²F1</b>={ext_metrics.get('q2f1', float('nan')):.4f}, "
            f"<b>Q²F2</b>={ext_metrics.get('q2f2', float('nan')):.4f}, "
            f"<b>Q²F3</b>={ext_metrics.get('q2f3', float('nan')):.4f}</li>"
        )
        rows.append(
            f"<li><b>r_m²(avg)</b>={ext_metrics.get('ext_rm2_avg', float('nan')):.4f}, "
            f"Δr_m²={ext_metrics.get('ext_rm2_delta', float('nan')):.4f} "
            f"(k={ext_metrics.get('ext_k', float('nan')):.3f}, k'={ext_metrics.get('ext_k_prime', float('nan')):.3f})</li>"
        )
        rows.append("</ul>")

    if perm_info is not None:
        rows.append("<h3>Y-randomization (permutation)</h3>")
        rows.append(
            f"Observed Q²={perm_info['q2_observed']:.4f} &nbsp;&nbsp; "
            f"p-value={perm_info['p_value']:.4g} (n={len(perm_info['q2_perm'])})<br>"
        )

    rows.append("<h3>Selected descriptors</h3>")
    rows.append("<pre>" + "\n".join(selected) + "</pre>")

    rows.append("<h3>Coefficients (OLS)</h3>")
    rows.append("<table border='1' cellspacing='0' cellpadding='4'>")
    rows.append("<tr><th>Term</th><th>β</th><th>SE</th><th>t</th><th>p</th><th>VIF</th></tr>")
    terms = ["Intercept"] + list(selected)
    for index, term in enumerate(terms):
        vif_val = "" if index == 0 else f"{vifs[index - 1]:.3g}"
        rows.append(
            f"<tr><td>{term}</td>"
            f"<td>{beta[index]:.6g}</td><td>{se[index]:.6g}</td><td>{t[index]:.6g}</td><td>{p[index]:.3g}</td>"
            f"<td>{vif_val}</td></tr>"
        )
    rows.append("</table>")

    return "\n".join(rows)
