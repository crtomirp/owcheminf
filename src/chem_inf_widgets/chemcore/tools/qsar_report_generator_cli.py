from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from chem_inf_widgets.chemcore.services.qsar_report_generator_service import (
    QSARReportConfig,
    generate_qsar_report,
    write_report_files,
)


def _read_optional(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a reproducible QSAR Studio report from QSAR output tables.")
    p.add_argument("--dataset", help="Optional dataset CSV/TSV.")
    p.add_argument("--metrics", help="Optional metrics CSV/TSV from QSAR Model Hub or Validation Dashboard.")
    p.add_argument("--predictions", help="Optional predictions CSV/TSV.")
    p.add_argument("--validation-summary", help="Optional validation summary CSV/TSV.")
    p.add_argument("--ad-summary", help="Optional applicability-domain summary CSV/TSV.")
    p.add_argument("--explanation-summary", help="Optional model-explanation summary CSV/TSV.")
    p.add_argument("--title", default="QSAR Studio Report")
    p.add_argument("--project-name", default="QSAR project")
    p.add_argument("--author", default="")
    p.add_argument("--out-prefix", default="qsar_report")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_qsar_report(
            dataset=_read_optional(args.dataset),
            metrics=_read_optional(args.metrics),
            predictions=_read_optional(args.predictions),
            validation_summary=_read_optional(args.validation_summary),
            ad_summary=_read_optional(args.ad_summary),
            explanation_summary=_read_optional(args.explanation_summary),
            config=QSARReportConfig(title=args.title, project_name=args.project_name, author=args.author),
        )
        paths = write_report_files(result, args.out_prefix)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("QSAR report generated.")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
