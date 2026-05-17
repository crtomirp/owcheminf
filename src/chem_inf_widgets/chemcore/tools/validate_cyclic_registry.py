from __future__ import annotations

"""CLI validator for the packaged cyclic registry.

Usage examples
--------------
python -m chem_inf_widgets.chemcore.tools.validate_cyclic_registry
python -m chem_inf_widgets.chemcore.tools.validate_cyclic_registry --json
python -m chem_inf_widgets.chemcore.tools.validate_cyclic_registry --collision-csv collisions.csv
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Sequence

from chem_inf_widgets.chemcore.descriptors.cyclic_registry_validation import (
    analyze_cyclic_registry,
    collision_rows,
    format_registry_report,
    report_to_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the packaged cyclic/heterocycle registry and report 4096-bit collision statistics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Analyze only the first N registry entries. Omit for full validation.",
    )
    parser.add_argument(
        "--no-smarts",
        action="store_true",
        help="Skip RDKit SMARTS compilation. Useful for metadata-only checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the human-readable report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all issues/collision examples in the human-readable report.",
    )
    parser.add_argument(
        "--max-collision-examples",
        type=int,
        default=200,
        help="Maximum number of collision bits retained in the report object.",
    )
    parser.add_argument(
        "--collision-csv",
        type=Path,
        default=None,
        help="Optional CSV file path for flattened collision examples.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path where the full JSON report should be written.",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit with status 1 if validation errors are found.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit with status 1 if validation warnings are found.",
    )
    return parser


def _write_collision_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["section", "bit", "collision_size", "entry_id", "name", "smarts"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    report = analyze_cyclic_registry(
        limit=args.limit,
        compile_smarts=not args.no_smarts,
        max_collision_examples=max(0, int(args.max_collision_examples)),
    )

    json_report = report_to_json(report, max_collision_examples=max(0, int(args.max_collision_examples)))

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json_report + "\n", encoding="utf-8")

    if args.collision_csv is not None:
        _write_collision_csv(args.collision_csv, collision_rows(report))

    if args.json:
        print(json_report)
    else:
        print(format_registry_report(report, verbose=bool(args.verbose)))
        if args.report_json is not None:
            print(f"\nJSON report written to: {args.report_json}")
        if args.collision_csv is not None:
            print(f"Collision CSV written to: {args.collision_csv}")

    if args.fail_on_errors and report.error_count > 0:
        return 1
    if args.fail_on_warnings and report.warning_count > 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
