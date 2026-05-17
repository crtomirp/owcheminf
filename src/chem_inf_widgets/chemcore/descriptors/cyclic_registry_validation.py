from __future__ import annotations

"""Validation and collision reporting for the cyclic registry fingerprint.

The code in this module is deliberately independent from the Orange widget.  It
is used by tests, by the command-line validator, and can also be imported in
notebooks when preparing a registry release or manuscript supplement.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from chem_inf_widgets.chemcore.descriptors.cyclic_registry_fingerprint import (
    BIT_SECTIONS,
    RegistryEntry,
    _bit_for_entry,
    _compiled_smarts,
    load_registry_entries,
)


@dataclass(frozen=True)
class RegistryIssue:
    """One validation issue found in the cyclic registry."""

    level: str
    code: str
    message: str
    entry_id: str = ""
    section: str = ""
    bit: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "entry_id": self.entry_id,
            "section": self.section,
            "bit": self.bit,
        }


@dataclass(frozen=True)
class SectionCollisionStats:
    """Collision statistics for one fingerprint section."""

    section: str
    bit_start: int
    bit_end: int
    width: int
    entries: int
    bits_used: int
    singleton_bits: int
    collision_bits: int
    colliding_entries: int
    max_entries_per_bit: int
    load_factor: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "section": self.section,
            "bit_start": self.bit_start,
            "bit_end": self.bit_end,
            "width": self.width,
            "entries": self.entries,
            "bits_used": self.bits_used,
            "singleton_bits": self.singleton_bits,
            "collision_bits": self.collision_bits,
            "colliding_entries": self.colliding_entries,
            "max_entries_per_bit": self.max_entries_per_bit,
            "load_factor": self.load_factor,
        }


@dataclass(frozen=True)
class RegistryCollision:
    """One fingerprint bit assigned to multiple registry entries."""

    section: str
    bit: int
    entry_ids: Tuple[str, ...]
    names: Tuple[str, ...]
    smarts: Tuple[str, ...]

    @property
    def collision_size(self) -> int:
        return len(self.entry_ids)

    def as_dict(self, *, max_items: int = 8) -> Dict[str, Any]:
        return {
            "section": self.section,
            "bit": self.bit,
            "collision_size": self.collision_size,
            "entry_ids": list(self.entry_ids[:max_items]),
            "names": list(self.names[:max_items]),
            "smarts": list(self.smarts[:max_items]),
            "truncated": self.collision_size > max_items,
        }


@dataclass(frozen=True)
class RegistryValidationReport:
    """Complete validation report for a cyclic registry release."""

    registry_version: str
    total_entries: int
    selected_entries: int
    section_stats: Tuple[SectionCollisionStats, ...]
    collisions: Tuple[RegistryCollision, ...]
    issues: Tuple[RegistryIssue, ...]
    group_counts: Mapping[str, int] = field(default_factory=dict)
    family_counts: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")

    def as_dict(self, *, include_collisions: bool = True, max_collision_examples: int = 25) -> Dict[str, Any]:
        data = {
            "registry_version": self.registry_version,
            "total_entries": self.total_entries,
            "selected_entries": self.selected_entries,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "group_counts": dict(self.group_counts),
            "family_counts_top25": dict(list(self.family_counts.items())[:25]),
            "section_stats": [s.as_dict() for s in self.section_stats],
            "issues": [i.as_dict() for i in self.issues],
            "metadata": dict(self.metadata),
        }
        if include_collisions:
            data["collisions"] = [c.as_dict() for c in self.collisions[:max_collision_examples]]
            data["collision_examples_truncated"] = len(self.collisions) > max_collision_examples
            data["collision_total_bits"] = len(self.collisions)
        return data


def _entry_key(entry: RegistryEntry) -> str:
    return f"{entry.section}|{entry.entry_id}|{entry.name}|{entry.smarts}"


def _section_for_bit(bit: int) -> str:
    for section, (start, end) in BIT_SECTIONS.items():
        if start <= int(bit) < end:
            return section
    return "out_of_range"


def _make_issue(level: str, code: str, message: str, entry: Optional[RegistryEntry] = None, bit: Optional[int] = None) -> RegistryIssue:
    return RegistryIssue(
        level=level,
        code=code,
        message=message,
        entry_id=getattr(entry, "entry_id", "") if entry is not None else "",
        section=getattr(entry, "section", "") if entry is not None else "",
        bit=bit,
    )


def _select_entries(entries: Sequence[RegistryEntry], limit: Optional[int]) -> List[RegistryEntry]:
    selected = list(entries)
    if limit is not None and int(limit) > 0:
        selected = selected[: int(limit)]
    return selected


def analyze_cyclic_registry(
    entries: Optional[Sequence[RegistryEntry]] = None,
    *,
    limit: Optional[int] = None,
    compile_smarts: bool = True,
    max_collision_examples: int = 200,
) -> RegistryValidationReport:
    """Validate registry entries and summarize bit collisions.

    Parameters
    ----------
    entries:
        Optional normalized entries.  If omitted, the packaged registry is used.
    limit:
        Optional maximum number of entries to analyze.  This is useful for quick
        widget tests, but full release validation should use ``None``.
    compile_smarts:
        If true, each SMARTS pattern is compiled with RDKit.
    max_collision_examples:
        Maximum number of collision objects retained in the report.  Section
        statistics are always computed on the full selected set.
    """
    if entries is None:
        registry_version, loaded = load_registry_entries()
        entries_all = list(loaded)
    else:
        registry_version = "custom"
        entries_all = list(entries)

    selected = _select_entries(entries_all, limit)
    issues: List[RegistryIssue] = []

    ids = [e.entry_id for e in selected]
    id_counts = Counter(ids)
    for entry in selected:
        if id_counts[entry.entry_id] > 1:
            issues.append(_make_issue("error", "duplicate_entry_id", f"Duplicate entry id: {entry.entry_id}", entry))
        if not entry.entry_id:
            issues.append(_make_issue("error", "missing_entry_id", "Missing entry id.", entry))
        if not entry.name:
            issues.append(_make_issue("warning", "missing_name", "Missing human-readable name.", entry))
        if not entry.smarts:
            issues.append(_make_issue("error", "missing_smarts", "Missing SMARTS pattern.", entry))

    seen_structural_keys: Dict[str, RegistryEntry] = {}
    for entry in selected:
        key = _entry_key(entry)
        if key in seen_structural_keys:
            issues.append(
                _make_issue(
                    "warning",
                    "duplicate_structural_key",
                    f"Duplicate section/id/name/SMARTS key also seen in {seen_structural_keys[key].entry_id}.",
                    entry,
                )
            )
        else:
            seen_structural_keys[key] = entry

    if compile_smarts:
        for entry in selected:
            if not entry.smarts:
                continue
            try:
                patt = _compiled_smarts(entry.smarts)
            except Exception as exc:
                issues.append(_make_issue("error", "smarts_exception", f"SMARTS compilation raised: {exc}", entry))
                continue
            if patt is None:
                issues.append(_make_issue("error", "invalid_smarts", f"Invalid SMARTS: {entry.smarts}", entry))

    bit_to_entries: Dict[int, List[RegistryEntry]] = defaultdict(list)
    for entry in selected:
        try:
            bit = _bit_for_entry(entry)
        except Exception as exc:
            issues.append(_make_issue("error", "bit_assignment_failed", f"Bit assignment failed: {exc}", entry))
            continue
        section = _section_for_bit(bit)
        if section == "out_of_range":
            issues.append(_make_issue("error", "bit_out_of_range", f"Assigned bit {bit} is outside the 4096-bit layout.", entry, bit))
        elif section != entry.section:
            issues.append(
                _make_issue(
                    "error",
                    "bit_section_mismatch",
                    f"Entry section {entry.section!r} assigned to bit {bit} in section {section!r}.",
                    entry,
                    bit,
                )
            )
        bit_to_entries[bit].append(entry)

    section_stats: List[SectionCollisionStats] = []
    for section, (start, end) in BIT_SECTIONS.items():
        if section in {"morgan", "ring_topology", "reserved"}:
            registry_entries = [e for e in selected if e.section == section]
        else:
            registry_entries = [e for e in selected if e.section == section]
        section_bits = {
            bit: es
            for bit, es in bit_to_entries.items()
            if start <= bit < end
        }
        bits_used = len(section_bits)
        collision_bits = sum(1 for es in section_bits.values() if len(es) > 1)
        colliding_entries = sum(len(es) for es in section_bits.values() if len(es) > 1)
        singleton_bits = sum(1 for es in section_bits.values() if len(es) == 1)
        max_entries_per_bit = max((len(es) for es in section_bits.values()), default=0)
        width = end - start
        section_stats.append(
            SectionCollisionStats(
                section=section,
                bit_start=start,
                bit_end=end,
                width=width,
                entries=len(registry_entries),
                bits_used=bits_used,
                singleton_bits=singleton_bits,
                collision_bits=collision_bits,
                colliding_entries=colliding_entries,
                max_entries_per_bit=max_entries_per_bit,
                load_factor=(len(registry_entries) / width) if width else 0.0,
            )
        )

    collisions_all: List[RegistryCollision] = []
    for bit, bit_entries in sorted(bit_to_entries.items()):
        if len(bit_entries) <= 1:
            continue
        collisions_all.append(
            RegistryCollision(
                section=_section_for_bit(bit),
                bit=int(bit),
                entry_ids=tuple(e.entry_id for e in bit_entries),
                names=tuple(e.name for e in bit_entries),
                smarts=tuple(e.smarts for e in bit_entries),
            )
        )

    group_counts = Counter((e.group or "unknown") for e in selected)
    family_counts = Counter((e.family or "unknown") for e in selected)
    metadata = {
        "bit_sections": {k: list(v) for k, v in BIT_SECTIONS.items()},
        "compile_smarts": bool(compile_smarts),
        "limit": limit,
        "collision_examples_retained": min(len(collisions_all), max_collision_examples),
        "collision_total_bits": len(collisions_all),
    }

    return RegistryValidationReport(
        registry_version=registry_version,
        total_entries=len(entries_all),
        selected_entries=len(selected),
        section_stats=tuple(section_stats),
        collisions=tuple(collisions_all[: int(max_collision_examples)]),
        issues=tuple(issues),
        group_counts=dict(group_counts),
        family_counts=dict(family_counts.most_common()),
        metadata=metadata,
    )


def format_registry_report(report: RegistryValidationReport, *, verbose: bool = False) -> str:
    """Format a registry validation report for terminal output."""
    lines: List[str] = []
    lines.append("Cyclic Registry Validation Report")
    lines.append("=" * 40)
    lines.append(f"Registry version: {report.registry_version}")
    lines.append(f"Entries analyzed: {report.selected_entries} / {report.total_entries}")
    lines.append(f"Errors: {report.error_count}")
    lines.append(f"Warnings: {report.warning_count}")
    lines.append("")
    lines.append("Section collision summary")
    lines.append("-" * 40)
    lines.append("section                         entries  width  bits_used  collision_bits  max/bit")
    for s in report.section_stats:
        lines.append(
            f"{s.section:<30} {s.entries:>7} {s.width:>6} {s.bits_used:>10} "
            f"{s.collision_bits:>15} {s.max_entries_per_bit:>8}"
        )
    lines.append("")
    lines.append("Group counts")
    lines.append("-" * 40)
    for group, n in sorted(report.group_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"{group}: {n}")
    if report.issues:
        lines.append("")
        lines.append("Issues")
        lines.append("-" * 40)
        for issue in report.issues[:50 if not verbose else len(report.issues)]:
            where = f" [{issue.entry_id}]" if issue.entry_id else ""
            bit = f" bit={issue.bit}" if issue.bit is not None else ""
            lines.append(f"{issue.level.upper()} {issue.code}{where}{bit}: {issue.message}")
        if not verbose and len(report.issues) > 50:
            lines.append(f"... {len(report.issues) - 50} additional issues omitted; use --verbose.")
    if report.collisions:
        lines.append("")
        lines.append("Collision examples")
        lines.append("-" * 40)
        for coll in report.collisions[:25 if not verbose else len(report.collisions)]:
            ids = ", ".join(coll.entry_ids[:5])
            if coll.collision_size > 5:
                ids += f", +{coll.collision_size - 5} more"
            lines.append(f"{coll.section} bit {coll.bit}: {coll.collision_size} entries -> {ids}")
        if not verbose and len(report.collisions) > 25:
            lines.append(f"... {len(report.collisions) - 25} additional collision examples omitted; use --verbose.")
    return "\n".join(lines)


def report_to_json(report: RegistryValidationReport, *, max_collision_examples: int = 100) -> str:
    return json.dumps(
        report.as_dict(include_collisions=True, max_collision_examples=max_collision_examples),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


def collision_rows(report: RegistryValidationReport, *, max_items_per_bit: int = 12) -> List[Dict[str, Any]]:
    """Return flattened collision rows suitable for CSV/TSV export."""
    rows: List[Dict[str, Any]] = []
    for coll in report.collisions:
        for i, entry_id in enumerate(coll.entry_ids[:max_items_per_bit]):
            rows.append(
                {
                    "section": coll.section,
                    "bit": coll.bit,
                    "collision_size": coll.collision_size,
                    "entry_id": entry_id,
                    "name": coll.names[i] if i < len(coll.names) else "",
                    "smarts": coll.smarts[i] if i < len(coll.smarts) else "",
                }
            )
    return rows
