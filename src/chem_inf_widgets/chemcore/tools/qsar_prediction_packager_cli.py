from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from chem_inf_widgets.chemcore.services.qsar_model_hub_service import QSARModelHubConfig, available_model_keys, train_qsar_model_hub
from chem_inf_widgets.chemcore.services.qsar_prediction_packager_service import (
    QSARPredictionPackagerConfig,
    load_model_pickle,
    predict_with_qsar_model,
    write_prediction_package,
)


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply a QSAR model to a query table and write a prediction package.")
    p.add_argument("query", help="Query CSV/TSV with descriptor/fingerprint columns.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-pickle", help="Pickled scikit-learn pipeline/model.")
    source.add_argument("--training-data", help="Training CSV/TSV; a model will be trained before prediction.")
    p.add_argument("--target-column", default="pActivity", help="Target column when --training-data is used.")
    p.add_argument("--id-column", default="compound_id")
    p.add_argument("--model", default="ridge", choices=available_model_keys(), help="Model key when --training-data is used.")
    p.add_argument("--prediction-column", default="predicted_pActivity")
    p.add_argument("--out-prefix", default="qsar_prediction_package")
    p.add_argument("--save-model", action="store_true", help="Also write the trained/loaded model pickle into the package.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        query = _read_table(args.query)
        if args.model_pickle:
            model = load_model_pickle(args.model_pickle)
        else:
            train = _read_table(args.training_data)
            train_result = train_qsar_model_hub(
                train,
                QSARModelHubConfig(target_column=args.target_column, id_column=args.id_column, model_key=args.model),
            )
            model = train_result.pipeline
        result = predict_with_qsar_model(
            model,
            query,
            QSARPredictionPackagerConfig(
                id_column=args.id_column,
                target_label=args.target_column,
                prediction_column=args.prediction_column,
            ),
        )
        paths = write_prediction_package(result, args.out_prefix, model=model if args.save_model else None)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("QSAR prediction package written.")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
