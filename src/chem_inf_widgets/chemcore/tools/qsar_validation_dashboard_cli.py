from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from chem_inf_widgets.chemcore.services.qsar_validation_dashboard_service import (
    QSARValidationConfig,
    validate_qsar_predictions,
    write_qsar_validation_outputs,
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate QSAR prediction results and produce residual diagnostics.")
    p.add_argument("input", help="Input CSV/TSV with observed and predicted columns.")
    p.add_argument("--observed-column", default="observed")
    p.add_argument("--predicted-column", default="predicted")
    p.add_argument("--split-column", default="split")
    p.add_argument("--id-column", default="compound_id")
    p.add_argument("--residual-threshold", type=float, default=None)
    p.add_argument("--z-threshold", type=float, default=3.0)
    p.add_argument("--out-prefix", default="qsar_validation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        df = _read_table(Path(args.input))
        cfg = QSARValidationConfig(
            observed_column=args.observed_column,
            predicted_column=args.predicted_column,
            split_column=args.split_column,
            id_column=args.id_column,
            residual_threshold=args.residual_threshold,
            z_threshold=args.z_threshold,
        )
        result = validate_qsar_predictions(df, cfg)
        paths = write_qsar_validation_outputs(result, args.out_prefix)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("QSAR Validation Dashboard finished.")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"overall_r2: {result.summary['overall_metrics'].get('r2')}")
    print(f"outliers: {result.summary['n_outliers']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
