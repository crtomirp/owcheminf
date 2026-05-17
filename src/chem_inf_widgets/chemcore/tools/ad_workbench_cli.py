from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chem_inf_widgets.chemcore.services.ad_workbench_service import (
    ADWorkbenchConfig,
    evaluate_applicability_domain_workbench,
    write_ad_workbench_outputs,
)


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(p, sep=sep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate QSAR applicability domain for reference/query tables.")
    parser.add_argument("reference_csv", help="Reference/training CSV or TSV table.")
    parser.add_argument("query_csv", nargs="?", help="Optional query/external CSV or TSV table. If omitted, reference is scored.")
    parser.add_argument("--id-column", default="compound_id")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--combine", choices=["and", "or"], default="and")
    parser.add_argument("--no-williams", action="store_true")
    parser.add_argument("--no-knn", action="store_true")
    parser.add_argument("--use-mahalanobis", action="store_true")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-quantile", type=float, default=0.95)
    parser.add_argument("--maha-alpha", type=float, default=0.95)
    parser.add_argument("--features", default="", help="Comma-separated feature columns. Default: auto-detect numeric columns.")
    args = parser.parse_args(argv)

    ref = _read_table(args.reference_csv)
    query = _read_table(args.query_csv) if args.query_csv else None
    features = tuple(x.strip() for x in args.features.split(",") if x.strip()) or None
    cfg = ADWorkbenchConfig(
        id_column=args.id_column,
        combine_mode=args.combine,
        use_williams=not args.no_williams,
        use_knn=not args.no_knn,
        use_mahalanobis=bool(args.use_mahalanobis),
        knn_k=int(args.knn_k),
        knn_quantile=float(args.knn_quantile),
        maha_alpha=float(args.maha_alpha),
        feature_columns=features,
    )
    result = evaluate_applicability_domain_workbench(ref, query, cfg)
    paths = write_ad_workbench_outputs(result, args.out_prefix)
    print("Applicability Domain Workbench completed.")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
