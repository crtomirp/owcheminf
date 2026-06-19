from __future__ import annotations

from typing import Any

from Orange.data import Table


def require_table(data: Table | None, widget=None, message: str = "No input data.") -> bool:
    if data is not None:
        return True
    if widget is not None and hasattr(widget, "_set_status"):
        try:
            widget._set_status(message, ok=False)
        except TypeError:
            widget._set_status(message)
    return False


def send_empty(output: Any) -> None:
    output.send(None)


def send_output_values(*pairs: tuple[Any, Any]) -> None:
    for output, value in pairs:
        output.send(value)


def clear_widget_outputs(widget: Any, *output_names: str) -> None:
    """Clear Orange output channels by sending ``None``.

    When an input or an important setting changes and a widget is not
    recomputed immediately, downstream widgets must not keep seeing stale
    results from the previous input. Passing no names clears all outputs
    declared on ``widget.Outputs``.
    """
    outputs = getattr(widget, "Outputs", None)
    if outputs is None:
        return
    names = output_names or tuple(
        name
        for name, value in vars(outputs).items()
        if not name.startswith("_") and hasattr(value, "send")
    )
    for name in names:
        output = getattr(outputs, name, None)
        if hasattr(output, "send"):
            output.send(None)
