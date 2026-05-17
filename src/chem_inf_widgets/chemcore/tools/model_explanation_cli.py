from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chem_inf_widgets.chemcore.services.model_explanation_service import (
    ModelExplanationConfig,
    explain_qsar_model,
    write_model_explanation_outputs,
)


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(p, sep=sep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain QSAR descriptor/fingerprint features from a table.")
    parser.add_argument("input_csv", help="Input CSV/TSV with target and numeric features.")
    parser.add_argument("--target-column", default="pActivity")
    parser.add_argument("--id-column", default="compound_id")
    parser.add_argument("--method", choices=["auto", "permutation", "univariate"], default="auto")
    parser.add_argument("--max-features", type=int, default=50)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args(argv)

    df = _read_table(args.input_csv)
    cfg = ModelExplanationConfig(
        target_column=args.target_column,
        id_column=args.id_column,
        method=args.method,
        max_features=int(args.max_features),
    )
    result = explain_qsar_model(df, model=None, config=cfg)
    paths = write_model_explanation_outputs(result, args.out_prefix)
    print("Model Explanation completed.")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
