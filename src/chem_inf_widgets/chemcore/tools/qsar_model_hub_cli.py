from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from chem_inf_widgets.chemcore.services.qsar_model_hub_service import (
    QSARModelHubConfig,
    available_model_keys,
    train_qsar_model_hub,
    write_qsar_model_hub_outputs,
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a QSAR regression model from a descriptor/fingerprint table.")
    p.add_argument("input", help="Input CSV/TSV with numeric feature columns and a target column.")
    p.add_argument("--target-column", default="pActivity")
    p.add_argument("--id-column", default="compound_id")
    p.add_argument("--model", default="random_forest", choices=available_model_keys())
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--scale", action="store_true", help="Force feature scaling.")
    p.add_argument("--no-scale", action="store_true", help="Disable feature scaling.")
    p.add_argument("--keep-constant", action="store_true", help="Do not remove constant features.")
    p.add_argument("--min-non-missing-fraction", type=float, default=0.70)
    p.add_argument("--out-prefix", default="qsar_model_hub")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scale = None
    if args.scale:
        scale = True
    if args.no_scale:
        scale = False
    try:
        df = _read_table(Path(args.input))
        cfg = QSARModelHubConfig(
            target_column=args.target_column,
            id_column=args.id_column,
            model_key=args.model,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            scale_features=scale,
            drop_constant_features=not args.keep_constant,
            min_non_missing_fraction=args.min_non_missing_fraction,
        )
        result = train_qsar_model_hub(df, cfg)
        paths = write_qsar_model_hub_outputs(result, args.out_prefix)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("QSAR Model Hub finished.")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print(f"test_r2: {result.test_metrics.get('test_r2')}")
    print(f"test_rmse: {result.test_metrics.get('test_rmse')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
