from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from chem_inf_widgets.chemcore.services.qsar_dataset_builder_service import (
    QSARDatasetBuilderConfig,
    build_qsar_dataset,
    read_records,
    write_result_files,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a QSAR-ready activity dataset from CSV/TSV records.",
    )
    p.add_argument("input", help="Input CSV/TSV file.")
    p.add_argument("--smiles-column", default=None)
    p.add_argument("--name-column", default=None)
    p.add_argument("--activity-column", default=None)
    p.add_argument("--unit-column", default=None)
    p.add_argument("--relation-column", default=None)
    p.add_argument("--endpoint-column", default=None)
    p.add_argument("--target-endpoint", default="")
    p.add_argument("--target-unit", default="nM")
    p.add_argument("--relation-policy", choices=["exact_only", "allow_inequalities"], default="exact_only")
    p.add_argument("--aggregation", choices=["median", "mean", "min", "max", "first"], default="median")
    p.add_argument("--duplicate-key", choices=["standard_inchikey", "canonical_smiles", "raw_smiles"], default="standard_inchikey")
    p.add_argument("--min-pactivity", type=float, default=None)
    p.add_argument("--max-pactivity", type=float, default=None)
    p.add_argument("--out-prefix", default="qsar_dataset")
    p.add_argument("--no-json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = QSARDatasetBuilderConfig(
        smiles_column=args.smiles_column,
        name_column=args.name_column,
        activity_column=args.activity_column,
        unit_column=args.unit_column,
        relation_column=args.relation_column,
        endpoint_column=args.endpoint_column,
        target_endpoint=args.target_endpoint,
        target_unit=args.target_unit,
        relation_policy=args.relation_policy,
        aggregation=args.aggregation,
        duplicate_key=args.duplicate_key,
        min_pactivity=args.min_pactivity,
        max_pactivity=args.max_pactivity,
    )
    records = read_records(args.input)
    result = build_qsar_dataset(records, config)
    files = write_result_files(result, args.out_prefix, write_json=not args.no_json)
    print("QSAR dataset builder summary")
    for key, value in result.summary.items():
        print(f"  {key}: {value}")
    print("Output files:")
    for key, path in files.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
