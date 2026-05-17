from __future__ import annotations

"""Command line interface for the Cyclic Registry Fingerprint.

This CLI exposes the same 4096-bit cyclic/heterocycle registry fingerprint used
by the Orange widget, but it can be run in scripts, notebooks, teaching labs, or
batch preprocessing workflows without opening Orange Canvas.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - availability depends on the runtime environment
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore

from chem_inf_widgets.chemcore.descriptors.cyclic_registry_fingerprint import (
    BIT_SECTIONS,
    DEFAULT_N_BITS,
    CyclicRegistryFingerprintResult,
    compute_cyclic_registry_fingerprints_from_smiles,
)


InputRows = Tuple[List[str], List[str], List[str]]  # ids, names, smiles


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute the 4096-bit Cyclic Registry Fingerprint from CSV, SMI/TXT, "
            "or SDF input without using Orange."
        )
    )
    p.add_argument("input", type=Path, help="Input file: .csv, .tsv, .smi, .smiles, .txt, .sdf, or .sd")
    p.add_argument("--out-prefix", type=Path, default=Path("cyclic_registry_fp"), help="Output path prefix.")
    p.add_argument("--format", choices=["auto", "csv", "tsv", "smi", "txt", "sdf"], default="auto")
    p.add_argument("--smiles-column", default="smiles", help="SMILES column for CSV/TSV input.")
    p.add_argument("--id-column", default=None, help="Optional ID column for CSV/TSV input.")
    p.add_argument("--name-column", default=None, help="Optional molecule name column for CSV/TSV input.")
    p.add_argument("--delimiter", default=None, help="CSV/TSV delimiter override. Default is inferred.")
    p.add_argument("--no-morgan", action="store_true", help="Disable the Morgan 0-2047 section.")
    p.add_argument("--no-sanitize", action="store_true", help="Parse SMILES/SDF without RDKit sanitization when possible.")
    p.add_argument("--no-atom-matches", action="store_true", help="Do not write atom match index tuples in the matches CSV.")
    p.add_argument("--max-registry-entries", type=int, default=None, help="Use only the first N registry entries; mainly for debugging/teaching.")
    p.add_argument(
        "--write-full-matrix",
        action="store_true",
        help="Write the full 4096-bit matrix CSV. Large files can become wide; by default only active bits are written.",
    )
    p.add_argument("--write-json", action="store_true", help="Write a compact JSON summary.")
    p.add_argument("--fail-on-invalid", action="store_true", help="Exit with status 2 if any input molecule fails parsing.")
    p.add_argument("--quiet", action="store_true", help="Suppress the human-readable summary printed to stdout.")
    return p


def _infer_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    ext = path.suffix.lower().lstrip(".")
    if ext in {"csv"}:
        return "csv"
    if ext in {"tsv"}:
        return "tsv"
    if ext in {"smi", "smiles"}:
        return "smi"
    if ext in {"sdf", "sd"}:
        return "sdf"
    return "txt"


def _read_delimited(path: Path, delimiter: str | None, smiles_column: str, id_column: str | None, name_column: str | None) -> InputRows:
    sep = delimiter if delimiter is not None else ("\t" if path.suffix.lower() == ".tsv" else ",")
    ids: List[str] = []
    names: List[str] = []
    smiles: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        field_lookup = {c.lower(): c for c in reader.fieldnames}
        smi_col = field_lookup.get(smiles_column.lower(), smiles_column)
        if smi_col not in reader.fieldnames:
            raise ValueError(f"SMILES column '{smiles_column}' not found. Available columns: {reader.fieldnames}")
        id_col = field_lookup.get(id_column.lower(), id_column) if id_column else None
        name_col = field_lookup.get(name_column.lower(), name_column) if name_column else None
        for i, row in enumerate(reader, start=1):
            smi = (row.get(smi_col) or "").strip()
            smiles.append(smi)
            ids.append((row.get(id_col) or str(i)).strip() if id_col else str(i))
            names.append((row.get(name_col) or ids[-1]).strip() if name_col else ids[-1])
    return ids, names, smiles


def _read_smi_or_txt(path: Path) -> InputRows:
    ids: List[str] = []
    names: List[str] = []
    smiles: List[str] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row_number = len(smiles) + 1
            parts = line.split()
            smi = parts[0]
            name = " ".join(parts[1:]) if len(parts) > 1 else str(row_number)
            ids.append(str(row_number))
            names.append(name)
            smiles.append(smi)
    return ids, names, smiles


def _read_sdf(path: Path, sanitize: bool) -> InputRows:
    if Chem is None:
        raise ImportError("RDKit is required to read SDF input.")
    ids: List[str] = []
    names: List[str] = []
    smiles: List[str] = []
    supplier = Chem.SDMolSupplier(str(path), sanitize=bool(sanitize), removeHs=False)
    for i, mol in enumerate(supplier, start=1):
        ids.append(str(i))
        if mol is None:
            names.append(str(i))
            smiles.append("")
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else str(i)
        names.append(name)
        try:
            smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        except Exception:
            smiles.append("")
    return ids, names, smiles


def read_input(args: argparse.Namespace) -> InputRows:
    path = args.input
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = _infer_format(path, args.format)
    if fmt in {"csv", "tsv"}:
        delimiter = args.delimiter
        if delimiter is None and fmt == "tsv":
            delimiter = "\t"
        return _read_delimited(path, delimiter, args.smiles_column, args.id_column, args.name_column)
    if fmt in {"smi", "txt"}:
        return _read_smi_or_txt(path)
    if fmt == "sdf":
        return _read_sdf(path, sanitize=not args.no_sanitize)
    raise ValueError(f"Unsupported input format: {fmt}")


def _write_active_bits(prefix: Path, ids: Sequence[str], names: Sequence[str], result: CyclicRegistryFingerprintResult) -> Path:
    out = prefix.with_suffix(".active_bits.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "input_id", "name", "canonical_smiles", "active_bit_count", "active_bits", "active_bit_names"])
        for out_row, input_idx in enumerate(result.valid_indices):
            active = np.flatnonzero(result.X[out_row]).astype(int).tolist()
            bit_names = [result.bit_names[b] if 0 <= b < len(result.bit_names) else f"bit_{b}" for b in active]
            writer.writerow([
                out_row,
                ids[input_idx] if input_idx < len(ids) else input_idx + 1,
                names[input_idx] if input_idx < len(names) else "",
                result.smiles[out_row],
                len(active),
                " ".join(str(b) for b in active),
                "|".join(bit_names),
            ])
    return out


def _write_matches(prefix: Path, ids: Sequence[str], names: Sequence[str], result: CyclicRegistryFingerprintResult) -> Path:
    out = prefix.with_suffix(".matches.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row", "input_index", "input_id", "name", "canonical_smiles", "bit", "entry_id", "entry_name",
            "section", "family", "smarts", "match_count", "atom_matches",
        ])
        for m in result.matches:
            input_idx = result.valid_indices[m.row] if m.row < len(result.valid_indices) else m.row
            atom_matches = ";".join("(" + ",".join(str(a) for a in match) + ")" for match in m.atom_matches)
            writer.writerow([
                m.row,
                input_idx,
                ids[input_idx] if input_idx < len(ids) else input_idx + 1,
                names[input_idx] if input_idx < len(names) else "",
                result.smiles[m.row] if m.row < len(result.smiles) else "",
                m.bit,
                m.entry_id,
                m.name,
                m.section,
                m.family,
                m.smarts,
                m.match_count,
                atom_matches,
            ])
    return out


def _write_failed(prefix: Path, ids: Sequence[str], names: Sequence[str], smiles: Sequence[str], result: CyclicRegistryFingerprintResult) -> Path:
    out = prefix.with_suffix(".failed.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input_index", "input_id", "name", "smiles"])
        for idx in result.failed_indices:
            writer.writerow([
                idx,
                ids[idx] if idx < len(ids) else idx + 1,
                names[idx] if idx < len(names) else "",
                smiles[idx] if idx < len(smiles) else "",
            ])
    return out


def _write_full_matrix(prefix: Path, ids: Sequence[str], names: Sequence[str], result: CyclicRegistryFingerprintResult) -> Path:
    out = prefix.with_suffix(".matrix.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "input_id", "name", "canonical_smiles"] + result.bit_names)
        for out_row, input_idx in enumerate(result.valid_indices):
            writer.writerow([
                out_row,
                ids[input_idx] if input_idx < len(ids) else input_idx + 1,
                names[input_idx] if input_idx < len(names) else "",
                result.smiles[out_row],
                *result.X[out_row].astype(int).tolist(),
            ])
    return out


def _section_counts(result: CyclicRegistryFingerprintResult) -> dict:
    counts = {name: 0 for name in BIT_SECTIONS}
    if result.X.size == 0:
        return counts
    active = np.flatnonzero(result.X.sum(axis=0)).astype(int).tolist()
    for bit in active:
        for section, (start, end) in BIT_SECTIONS.items():
            if start <= bit < end:
                counts[section] += 1
                break
    return counts


def _write_summary(prefix: Path, args: argparse.Namespace, ids: Sequence[str], result: CyclicRegistryFingerprintResult, written: Sequence[Path]) -> Path:
    out = prefix.with_suffix(".summary.json")
    summary = {
        "input": str(args.input),
        "out_prefix": str(prefix),
        "molecules_input": len(ids),
        "molecules_valid": len(result.valid_indices),
        "molecules_failed": len(result.failed_indices),
        "matches": len(result.matches),
        "n_bits": result.n_bits,
        "fingerprint_version": result.fingerprint_version,
        "registry_version": result.registry_version,
        "params": result.params,
        "bit_sections": {k: list(v) for k, v in BIT_SECTIONS.items()},
        "active_section_bit_counts": _section_counts(result),
        "errors": result.errors[:200],
        "outputs": [str(p) for p in written],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ids, names, smiles = read_input(args)
        result = compute_cyclic_registry_fingerprints_from_smiles(
            smiles,
            include_morgan=not args.no_morgan,
            sanitize=not args.no_sanitize,
            max_registry_entries=args.max_registry_entries,
            include_atom_matches=not args.no_atom_matches,
        )
        prefix = args.out_prefix
        written: List[Path] = []
        written.append(_write_active_bits(prefix, ids, names, result))
        written.append(_write_matches(prefix, ids, names, result))
        written.append(_write_failed(prefix, ids, names, smiles, result))
        if args.write_full_matrix:
            written.append(_write_full_matrix(prefix, ids, names, result))
        if args.write_json:
            written.append(_write_summary(prefix, args, ids, result, written))
        if not args.quiet:
            print("Cyclic Registry Fingerprint CLI")
            print(f"Input molecules: {len(ids)}")
            print(f"Valid molecules: {len(result.valid_indices)}")
            print(f"Failed molecules: {len(result.failed_indices)}")
            print(f"Registry matches: {len(result.matches)}")
            print(f"Fingerprint bits: {DEFAULT_N_BITS}")
            for p in written:
                print(f"Wrote: {p}")
            if result.errors:
                print(f"Warnings/errors recorded: {len(result.errors)}; see summary JSON with --write-json for details.")
        if args.fail_on_invalid and result.failed_indices:
            return 2
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
