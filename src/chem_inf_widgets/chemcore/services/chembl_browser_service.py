from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence

from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.from_orange import table_to_chemmols_with_report
from chem_inf_widgets.widgets.ui_helpers import format_table_report

from .chembl_models import ChemBLBioactivityRecord, ChemBLTargetRecord


@dataclass(frozen=True)
class SummaryRow:
    key: str
    count: int
    pct: float


def summarize_activity_types(
    records: List[ChemBLBioactivityRecord],
    top_n: int = 12,
) -> List[SummaryRow]:
    total = max(1, len(records))
    keys: List[str] = []
    for record in records:
        standard_type = (getattr(record, "standard_type", "") or "").strip()
        standard_units = (getattr(record, "standard_units", "") or "").strip()
        if standard_type and standard_units:
            keys.append(f"{standard_type} ({standard_units})")
        elif standard_type:
            keys.append(standard_type)
        else:
            keys.append("UNKNOWN")
    counts = Counter(keys)
    return [SummaryRow(key, count, 100.0 * count / total) for key, count in counts.most_common(top_n)]


def format_number(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return ""


def compile_user_pattern(text: str) -> Optional[re.Pattern]:
    """
    Supports:
      - /regex/
      - wildcards: * ?
      - plain substring (escaped)
    """
    value = (text or "").strip()
    if not value:
        return None

    if len(value) >= 2 and value.startswith("/") and value.endswith("/"):
        body = value[1:-1]
        try:
            return re.compile(body, re.IGNORECASE)
        except re.error:
            return None

    if "*" in value or "?" in value:
        escaped = re.escape(value).replace(r"\*", ".*").replace(r"\?", ".")
        try:
            return re.compile(escaped, re.IGNORECASE)
        except re.error:
            return None

    try:
        return re.compile(re.escape(value), re.IGNORECASE)
    except re.error:
        return None


def target_search_blob(target: ChemBLTargetRecord) -> str:
    return " | ".join(
        [
            target.chembl_id or "",
            target.pref_name or "",
            target.organism or "",
            target.target_type or "",
        ]
    )


def query_needs_postfilter(query: str) -> bool:
    return (
        ("*" in query)
        or ("?" in query)
        or (query.startswith("/") and query.endswith("/"))
    )


def filter_targets(
    targets: Sequence[ChemBLTargetRecord],
    pattern: Optional[re.Pattern],
) -> List[ChemBLTargetRecord]:
    if pattern is None:
        return list(targets)
    return [target for target in targets if pattern.search(target_search_blob(target))]


def format_output_summary(table, mols: Sequence[ChemMol], *, selected: bool = False) -> str:
    prefix = "Selected" if selected else "Output"
    n_rows = 0 if table is None else len(table)
    n_mols = len(mols or [])

    if table is None:
        return f"{prefix}: 0 rows, {n_mols} molecules."

    try:
        _parsed, report = table_to_chemmols_with_report(table)
        return f"{format_table_report(report, prefix=prefix, valid_label='valid SMILES')}, molecules={n_mols}"
    except Exception:
        return f"{prefix}: {n_rows} rows, {n_mols} molecules."
