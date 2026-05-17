from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from chem_inf_widgets.chemcore.services.molecule_import_service import (
    MoleculeImportConfig,
    import_molecule_file,
    import_records_as_dicts,
    import_summary_as_rows,
)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="owcheminf-molecule-import",
        description="Import molecules from CSV/TSV/SMI/TXT/SDF and write standardized import tables/reports.",
    )
    parser.add_argument("input", help="Input file (.csv, .tsv, .smi, .smiles, .txt, .sdf, .sd).")
    parser.add_argument("--smiles-column", default=None, help="SMILES column for CSV/TSV input. Auto-detected if omitted.")
    parser.add_argument("--name-column", default=None, help="Name/identifier column for CSV/TSV input. Auto-detected if omitted.")
    parser.add_argument("--delimiter", default=None, help="Optional delimiter for table input. Use '\\t' for tab.")
    parser.add_argument("--no-sanitize", action="store_true", help="Disable RDKit sanitization during import.")
    parser.add_argument("--keep-hs", action="store_true", help="Keep explicit hydrogens where possible.")
    parser.add_argument("--out-prefix", default="molecule_import", help="Output prefix, without extension.")
    parser.add_argument("--json", action="store_true", help="Also write JSON summary/report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = MoleculeImportConfig(
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        delimiter=args.delimiter,
        sanitize=not bool(args.no_sanitize),
        remove_hs=not bool(args.keep_hs),
    )
    result = import_molecule_file(args.input, cfg)
    prefix = Path(args.out_prefix)
    record_rows = import_records_as_dicts(result.records)
    valid_rows = [r for r in record_rows if str(r.get("ok")) in {"1", "True", "true"}]
    failed_rows = [r for r in record_rows if str(r.get("ok")) not in {"1", "True", "true"}]
    _write_csv(prefix.with_suffix(".import_report.csv"), record_rows)
    _write_csv(prefix.with_suffix(".molecules.csv"), valid_rows)
    _write_csv(prefix.with_suffix(".failed.csv"), failed_rows)
    _write_csv(prefix.with_suffix(".summary.csv"), import_summary_as_rows(result.summary))
    if args.json:
        payload = {
            "summary": result.summary.__dict__,
            "records": record_rows,
        }
        prefix.with_suffix(".summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Imported {result.summary.valid_records}/{result.summary.total_records} records "
        f"({result.summary.failed_records} failed). Outputs written with prefix: {prefix}"
    )
    return 0 if result.summary.failed_records == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
