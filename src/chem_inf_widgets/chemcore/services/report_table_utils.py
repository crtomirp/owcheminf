from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from Orange.data import ContinuousVariable, Domain, StringVariable, Table

from .orange_table_utils import records_to_orange_table, safe_table_from_numpy


def _empty_table(
    *,
    numeric_columns: Sequence[str],
    meta_columns: Sequence[str],
    name: str,
) -> Table:
    attrs = [ContinuousVariable(str(col)) for col in numeric_columns]
    metas = [StringVariable(str(col)) for col in meta_columns]
    return safe_table_from_numpy(
        Domain(attrs, metas=metas),
        X=np.empty((0, len(attrs)), dtype=float),
        metas=np.empty((0, len(metas)), dtype=object),
        name=name,
    )


def report_rows_to_table(
    rows: Iterable[Mapping[str, Any]],
    *,
    numeric_columns: Sequence[str] = (),
    meta_columns: Sequence[str] = (),
    name: str = "Report",
) -> Table:
    rows_list = [dict(row) for row in rows]
    numeric_columns = [str(col) for col in numeric_columns]
    meta_columns = [str(col) for col in meta_columns]
    if not rows_list:
        return _empty_table(
            numeric_columns=numeric_columns,
            meta_columns=meta_columns,
            name=name,
        )

    table = records_to_orange_table(
        rows_list,
        attribute_columns=numeric_columns or None,
        meta_columns=meta_columns or None,
        numeric_as_attributes=True,
        name=name,
    )
    if table is not None:
        return table
    return _empty_table(
        numeric_columns=numeric_columns,
        meta_columns=meta_columns,
        name=name,
    )


def summary_rows_to_table(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric_column: str = "metric",
    value_column: str = "value",
    description_column: str = "description",
    name: str = "Summary",
) -> Table:
    rows_list = [dict(row) for row in rows]
    present_descriptions = any(description_column in row for row in rows_list)
    numeric_value = rows_list and all(
        isinstance(row.get(value_column), (int, float, np.integer, np.floating))
        and not isinstance(row.get(value_column), bool)
        for row in rows_list
        if row.get(value_column) not in (None, "")
    )
    numeric_columns = [value_column] if numeric_value else []
    meta_columns = [metric_column]
    if present_descriptions:
        meta_columns.append(description_column)
    if value_column not in numeric_columns:
        meta_columns.append(value_column)

    return report_rows_to_table(
        rows_list,
        numeric_columns=numeric_columns,
        meta_columns=meta_columns,
        name=name,
    )


__all__ = [
    "report_rows_to_table",
    "summary_rows_to_table",
]
