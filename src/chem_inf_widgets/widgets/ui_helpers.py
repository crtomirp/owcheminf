from __future__ import annotations

from typing import Any, Optional, Sequence

from chem_inf_widgets.chemcore.services.from_orange import TableMolConversionReport


def format_waiting_status(subject: str = "input") -> str:
    return f"Waiting for {subject}…"


def format_no_input_status(subject: str = "input") -> str:
    if subject == "input":
        return "No input."
    return f"No {subject}."


def format_error_status(message: str) -> str:
    return f"Error: {message}"


def format_failed_status(message: str) -> str:
    return f"Failed: {message}"


def format_done_status(*parts: str, prefix: str = "Done") -> str:
    clean_parts = [part for part in parts if part]
    if not clean_parts:
        return prefix
    return f"{prefix}: " + ", ".join(clean_parts)


def format_required_inputs_status(*subjects: str) -> str:
    clean = [subject for subject in subjects if subject]
    if not clean:
        return "Required inputs are missing."
    if len(clean) == 1:
        return f"{clean[0]} is required."
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]} are required."
    return f"{', '.join(clean[:-1])}, and {clean[-1]} are required."


def format_loaded_status(
    count: int,
    *,
    item_label: str = "rows",
    prefix: str = "Input loaded",
) -> str:
    return f"{prefix}: {count} {item_label}."


def format_result_count_status(
    count: int,
    *,
    item_label: str,
    prefix: str = "Found",
) -> str:
    return f"{prefix}: {count} {item_label}."


def format_skip_warning(
    count: int,
    *,
    subject: str = "input rows",
    action: str = "were skipped",
) -> Optional[str]:
    if count <= 0:
        return None
    return f"{count} {subject} {action}."


def format_table_report(
    report: TableMolConversionReport,
    *,
    prefix: str = "Input",
    row_label: str = "rows",
    valid_label: str = "valid",
    invalid_label: str = "invalid",
    include_smiles_column: bool = True,
    column_label: str = "column",
) -> str:
    parts = [
        f"{prefix}: {row_label}={report.n_rows}",
        f"{valid_label}={report.n_valid}",
        f"{invalid_label}={report.n_invalid}",
    ]
    if include_smiles_column and report.smiles_column:
        parts.append(f"{column_label}={report.smiles_column}")
    return ", ".join(parts)


def format_conversion_report(
    report: TableMolConversionReport,
    *,
    prefix: str = "Loaded",
    item_label: str = "molecules",
    include_smiles_column: bool = True,
    column_label: str = "SMILES column",
) -> str:
    text = f"{prefix}: {report.n_valid}/{report.n_rows} {item_label}"
    if include_smiles_column and report.smiles_column:
        text += f" ({column_label}: '{report.smiles_column}')"
    return text + "."


def format_skipped_rows_warning(
    report: TableMolConversionReport,
    *,
    prefix: str = "Skipped rows",
    row_label: str = "Rows",
    max_rows: int = 5,
) -> Optional[str]:
    if report.n_invalid <= 0:
        return None
    preview_rows = ", ".join(map(str, report.skipped_rows[:max_rows]))
    suffix = ", ..." if len(report.skipped_rows) > max_rows else ""
    return f"{prefix}: {report.n_invalid}. {row_label}: {preview_rows}{suffix}"


def set_widget_warning(widget: Any, message: Optional[str]) -> None:
    widget.warning(message or "")


def set_widget_error(widget: Any, message: Optional[str]) -> None:
    widget.error(message or "")


def set_widget_information(widget: Any, message: Optional[str]) -> None:
    widget.information(message or "")


def clear_widget_messages(
    widget: Any,
    *,
    warning: bool = True,
    error: bool = True,
    information: bool = False,
) -> None:
    if warning:
        set_widget_warning(widget, "")
    if error:
        set_widget_error(widget, "")
    if information:
        set_widget_information(widget, "")
