from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from chem_inf_widgets.chemcore.services.molecule_qc_service import (
    MoleculeQCConfig,
    qc_records_as_dicts,
    qc_summary_as_rows,
    run_molecule_qc,
)


def _read_csv_like(path: Path, smiles_column: str, delimiter: str | None = None) -> tuple[List[str], List[Dict[str, Any]]]:
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Input file has no header.")
        if smiles_column not in reader.fieldnames:
            raise ValueError(f"SMILES column '{smiles_column}' not found. Available columns: {reader.fieldnames}")
        rows = list(reader)
    return [str(row.get(smiles_column, "") or "") for row in rows], rows


def _read_smi(path: Path) -> tuple[List[str], List[Dict[str, Any]]]:
    smiles: List[str] = []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smi = parts[0]
            name = " ".join(parts[1:]) if len(parts) > 1 else ""
            smiles.append(smi)
            rows.append({"smiles": smi, "name": name})
    return smiles, rows


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": result.summary.version,
        "summary": {row["metric"]: row["value"] for row in qc_summary_as_rows(result.summary)},
        "issue_counts": result.summary.issue_counts,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run molecule quality-control checks on CSV/TSV/SMI input files.")
    p.add_argument("input", help="Input .csv, .tsv, .smi, .smiles, or .txt file.")
    p.add_argument("--smiles-column", default="smiles", help="SMILES column for CSV/TSV input. Default: smiles")
    p.add_argument("--out-prefix", default="molecule_qc", help="Output prefix. Default: molecule_qc")
    p.add_argument("--duplicate-key", choices=["canonical_smiles", "inchikey"], default="canonical_smiles")
    p.add_argument("--max-mw", type=float, default=900.0)
    p.add_argument("--min-heavy-atoms", type=int, default=3)
    p.add_argument("--max-heavy-atoms", type=int, default=90)
    p.add_argument("--max-fragments", type=int, default=1)
    p.add_argument("--allow-charged", action="store_true", help="Do not flag non-zero net formal charge.")
    p.add_argument("--allow-metals", action="store_true", help="Do not flag metals/metalloids.")
    p.add_argument("--allow-isotopes", action="store_true", help="Do not flag isotope labels.")
    p.add_argument("--allow-radicals", action="store_true", help="Do not flag radicals.")
    p.add_argument("--ignore-chiral-stereo", action="store_true", help="Do not flag unassigned chiral centers.")
    p.add_argument("--ignore-double-bond-stereo", action="store_true", help="Do not flag unassigned E/Z double-bond stereo.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    suffix = input_path.suffix.lower()
    if suffix in {".csv", ".tsv", ".tab"}:
        smiles, _rows = _read_csv_like(input_path, args.smiles_column)
    elif suffix in {".smi", ".smiles", ".txt"}:
        smiles, _rows = _read_smi(input_path)
    else:
        raise SystemExit(f"Unsupported input format: {suffix}. Use CSV/TSV/SMI/TXT.")

    cfg = MoleculeQCConfig(
        duplicate_key=args.duplicate_key,
        max_mw=args.max_mw,
        min_heavy_atoms=args.min_heavy_atoms,
        max_heavy_atoms=args.max_heavy_atoms,
        max_fragments=args.max_fragments,
        flag_metals=not args.allow_metals,
        flag_isotopes=not args.allow_isotopes,
        flag_radicals=not args.allow_radicals,
        flag_formal_charge=not args.allow_charged,
        flag_missing_chiral_stereo=not args.ignore_chiral_stereo,
        flag_missing_double_bond_stereo=not args.ignore_double_bond_stereo,
    )
    result = run_molecule_qc(smiles, cfg)
    prefix = Path(args.out_prefix)
    report_rows = qc_records_as_dicts(result.records)
    clean_rows = [report_rows[i] for i in result.clean_indices]
    problem_rows = [report_rows[i] for i in result.problem_indices]
    _write_csv(prefix.with_suffix(".qc_report.csv"), report_rows)
    _write_csv(prefix.with_suffix(".clean.csv"), clean_rows)
    _write_csv(prefix.with_suffix(".problems.csv"), problem_rows)
    _write_csv(prefix.with_suffix(".summary.csv"), qc_summary_as_rows(result.summary))
    _write_summary_json(prefix.with_suffix(".summary.json"), result)
    print(
        f"Molecule QC complete: total={result.summary.total}, clean={result.summary.clean}, "
        f"problem={result.summary.problem}, invalid={result.summary.invalid}, "
        f"duplicate_groups={result.summary.duplicate_groups}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
