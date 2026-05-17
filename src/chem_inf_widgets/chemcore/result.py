from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Generic, Iterable, Literal, Optional, TypeVar

T = TypeVar("T")
Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class ServiceIssue:
    """Structured issue emitted by chemcore services.

    The goal is to avoid silent ``except/pass`` behaviour. Services can return
    useful data and still preserve row-level warnings/errors for Orange widgets,
    reports, and tests.
    """

    code: str
    message: str
    severity: Severity = "warning"
    row_index: Optional[int] = None
    molecule_id: Optional[str] = None
    field: Optional[str] = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "row_index": "" if self.row_index is None else self.row_index,
            "molecule_id": self.molecule_id or "",
            "field": self.field or "",
            **{f"detail_{k}": v for k, v in self.details.items()},
        }


@dataclass(frozen=True)
class ServiceResult(Generic[T]):
    """Generic return object for robust cheminformatics services."""

    data: T
    issues: list[ServiceIssue] = dataclass_field(default_factory=list)
    summary: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def warnings(self) -> list[ServiceIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def errors(self) -> list[ServiceIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]


def count_issues(issues: Iterable[ServiceIssue]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue.code] = counts.get(issue.code, 0) + 1
    return dict(sorted(counts.items()))
