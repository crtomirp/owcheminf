import numpy as np
from AnyQt.QtCore import Qt as _Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QLabel, QTableWidget, QTableWidgetItem
from matplotlib.figure import Figure


def build_feature_message_label(html: str) -> QLabel:
    label = QLabel(html)
    label.setWordWrap(True)
    label.setStyleSheet("padding:12px; color:#555;")
    return label


def build_feature_subtitle_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-size:10px; color:#6b7280; padding:2px 0;")
    return label


def build_feature_chart_figure(
    chart_names,
    chart_values,
    chart_colors,
    *,
    value_label: str,
    chart_title: str,
) -> Figure:
    top_n = len(chart_names)
    y_pos = np.arange(top_n)
    fig = Figure(figsize=(7, max(3, top_n * 0.28)))
    ax = fig.add_subplot(111)
    ax.barh(y_pos, chart_values, color=list(chart_colors), edgecolor="none", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(chart_names), fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel(value_label, fontsize=9)
    ax.set_title(chart_title, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(pad=1.0)
    return fig


def build_features_table(
    names,
    values,
    col_label: str,
    *,
    ses=None,
    ts=None,
    ps=None,
    vifs=None,
) -> QTableWidget:
    has_stats = ses is not None and ts is not None and ps is not None
    has_vif = vifs is not None
    headers = ["Rank", "Descriptor", col_label]
    if has_stats:
        headers += ["SE", "t", "p-value"]
    if has_vif:
        headers.append("VIF")

    table = QTableWidget(len(names), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setStretchLastSection(True)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setAlternatingRowColors(True)

    for i, name in enumerate(names):
        rank_item = QTableWidgetItem()
        rank_item.setData(_Qt.DisplayRole, i + 1)
        table.setItem(i, 0, rank_item)
        table.setItem(i, 1, QTableWidgetItem(name))

        val = float(values[i])
        val_item = QTableWidgetItem()
        if np.isfinite(val):
            val_item.setData(_Qt.DisplayRole, round(val, 6))
            if col_label == "Coefficient":
                val_item.setBackground(QColor("#dcfce7" if val >= 0 else "#fee2e2"))
        else:
            val_item.setText("—")
        table.setItem(i, 2, val_item)

        col = 3
        if has_stats:
            se_item = QTableWidgetItem()
            se_item.setData(_Qt.DisplayRole, round(float(ses[i]), 5))
            table.setItem(i, col, se_item)
            col += 1

            t_item = QTableWidgetItem()
            t_item.setData(_Qt.DisplayRole, round(float(ts[i]), 4))
            table.setItem(i, col, t_item)
            col += 1

            p = float(ps[i])
            p_item = QTableWidgetItem()
            p_item.setData(_Qt.DisplayRole, round(p, 5))
            p_bg = "#dcfce7" if p < 0.01 else "#d1fae5" if p < 0.05 else "#fef3c7" if p < 0.10 else "#fee2e2"
            p_item.setBackground(QColor(p_bg))
            table.setItem(i, col, p_item)
            col += 1

        if has_vif:
            vif_val = float(vifs[i]) if i < len(vifs) else float("nan")
            vif_item = QTableWidgetItem()
            if np.isfinite(vif_val):
                vif_item.setData(_Qt.DisplayRole, round(vif_val, 3))
                vif_bg = "#dcfce7" if vif_val < 5 else "#fef3c7" if vif_val < 10 else "#fee2e2"
                vif_item.setBackground(QColor(vif_bg))
            else:
                vif_item.setText("—")
            table.setItem(i, col, vif_item)

    table.resizeColumnToContents(0)
    table.resizeColumnToContents(1)
    table.setSortingEnabled(True)
    return table
