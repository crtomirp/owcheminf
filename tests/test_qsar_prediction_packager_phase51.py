import numpy as np
import pandas as pd
from pathlib import Path
from Orange.widgets.tests.base import WidgetTest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
import pytest
from types import SimpleNamespace

from chem_inf_widgets.chemcore.descriptors.fingerprints import (
    compute_fingerprints_from_smiles,
)
from chem_inf_widgets.chemcore.mol import ChemMol
from chem_inf_widgets.chemcore.services.mordred_descriptor_service import (
    MORDRED_AVAILABLE,
    MordredComputeConfig,
    MordredDescriptorService,
)
from chem_inf_widgets.chemcore.services.qsar_prediction_packager_service import (
    QSARPredictionPackagerConfig,
    QSARPredictionModelBundle,
    build_qsar_prediction_bundle,
    load_model_pickle,
    predict_with_qsar_model,
    selected_feature_names_from_model,
    write_model_bundle_package,
)
from chem_inf_widgets.chemcore.services.qsar_regression_service import (
    RDKit_DESCRIPTOR_NAMES,
    _rdkit_descriptor_row,
)
from chem_inf_widgets.widgets.ow_qsar_prediction_packager import (
    OWQSARPredictionPackager,
)


def _rdkit_training_frame(smiles_list: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [_rdkit_descriptor_row(smiles) for smiles in smiles_list],
        columns=RDKit_DESCRIPTOR_NAMES,
    )


def test_prediction_packager_auto_computes_rdkit_descriptors_from_smiles():
    train_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCO", "CCCl"]
    X_train = _rdkit_training_frame(train_smiles)
    y_train = 0.02 * X_train["MolWt"] + 0.40 * X_train["MolLogP"]
    model = LinearRegression().fit(X_train, y_train)
    bundle = build_qsar_prediction_bundle(
        model,
        feature_names=list(X_train.columns),
        target_label="demo_activity",
        recipe_kind="rdkit_compact",
    )

    query = pd.DataFrame(
        {
            "compound_id": ["Q1", "Q2", "Q3"],
            "canonical_smiles": ["CCO", "CCCN", "c1ccncc1"],
        }
    )
    result = predict_with_qsar_model(
        bundle,
        query,
        QSARPredictionPackagerConfig(
            id_column="compound_id",
            target_label="demo_activity",
        ),
    )

    assert len(result.predictions) == 3
    assert result.predictions["predicted_pActivity"].notna().all()
    assert result.package_manifest["recipe_kind"] == "rdkit_compact"
    assert bool(result.package_manifest["auto_feature_engineering_used"]) is True
    assert result.package_manifest["rows_failed"] == 0


def test_prediction_packager_auto_computes_descriptor_and_fingerprint_recipe():
    train_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCO", "CCCl", "CCBr"]
    desc_df = _rdkit_training_frame(train_smiles)[["MolWt", "MolLogP"]]
    fp_res = compute_fingerprints_from_smiles(
        train_smiles,
        fp_type="morgan",
        bit_size=16,
        radius=2,
        remove_low_variance=False,
    )
    fp_df = pd.DataFrame(fp_res.X, columns=fp_res.bit_names)
    feature_df = pd.concat(
        [desc_df, fp_df[["morgan_0001", "morgan_0007"]]],
        axis=1,
    )
    y_train = (
        0.03 * feature_df["MolWt"]
        + 0.50 * feature_df["MolLogP"]
        + 0.80 * feature_df["morgan_0001"]
        - 0.35 * feature_df["morgan_0007"]
    )
    model = Ridge(alpha=0.1).fit(feature_df, y_train)
    bundle = build_qsar_prediction_bundle(
        model,
        feature_names=list(feature_df.columns),
        target_label="hybrid_activity",
        recipe_kind="rdkit_compact_plus_fingerprint",
        fingerprint_type="morgan",
        fingerprint_radius=2,
        fingerprint_n_bits=16,
    )

    query = pd.DataFrame(
        {
            "compound_id": ["H1", "H2"],
            "SMILES": ["CCO", "c1ccncc1"],
        }
    )
    result = predict_with_qsar_model(
        bundle,
        query,
        QSARPredictionPackagerConfig(
            id_column="compound_id",
            target_label="hybrid_activity",
        ),
    )

    assert len(result.predictions) == 2
    assert result.predictions["predicted_pActivity"].notna().all()
    assert result.package_manifest["recipe_kind"] == "rdkit_compact_plus_fingerprint"
    assert result.package_manifest["rows_failed"] == 0
    assert set(result.feature_report["feature"]) == set(feature_df.columns)


def test_prediction_packager_supports_legacy_source_prefixed_feature_names():
    train_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCO", "CCCl"]
    X_train = _rdkit_training_frame(train_smiles)[["MolWt", "MolLogP"]].copy()
    X_train.columns = ["source_MolWt", "source_MolLogP"]
    y_train = 0.02 * X_train["source_MolWt"] + 0.40 * X_train["source_MolLogP"]
    model = LinearRegression().fit(X_train, y_train)
    bundle = build_qsar_prediction_bundle(
        model,
        feature_names=list(X_train.columns),
        target_label="legacy_activity",
    )

    query = pd.DataFrame(
        {
            "compound_id": ["L1", "L2"],
            "canonical_smiles": ["CCO", "c1ccncc1"],
        }
    )
    result = predict_with_qsar_model(
        bundle,
        query,
        QSARPredictionPackagerConfig(
            id_column="compound_id",
            target_label="legacy_activity",
        ),
    )

    assert len(result.predictions) == 2
    assert result.predictions["predicted_pActivity"].notna().all()
    assert result.package_manifest["recipe_kind"] == "rdkit_compact"
    assert "source_MolWt" in result.package_manifest["feature_names"]
    assert result.package_manifest["rows_failed"] == 0
    assert result.package_manifest["bundle_model_name"] == "LinearRegression"
    assert result.package_manifest["bundle_feature_count"] == 2


@pytest.mark.skipif(not MORDRED_AVAILABLE, reason="mordred is not installed")
def test_prediction_packager_auto_computes_mordred_descriptors_from_smiles():
    service = MordredDescriptorService(MordredComputeConfig(ignore_3d=True, nproc=1))
    train_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCO", "CCCl", "CCBr", "c1ccncc1"]
    mols_maybe, valid_idx = service.smiles_to_mols(train_smiles)
    valid_mols = [mols_maybe[i] for i in valid_idx if mols_maybe[i] is not None]

    candidate_names = [info.name for info in service.list_descriptors()[:80]]
    mordred_valid = service.compute(valid_mols, candidate_names)
    mordred_full = service.df_to_full_length(mordred_valid, valid_idx, len(train_smiles))
    usable_cols = [
        col
        for col in mordred_full.columns
        if np.isfinite(pd.to_numeric(mordred_full[col], errors="coerce")).sum() >= 6
        and pd.to_numeric(mordred_full[col], errors="coerce").nunique(dropna=True) >= 3
    ][:4]
    assert usable_cols, "No stable Mordred descriptor subset found for test."

    X_train = mordred_full[usable_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train = np.linspace(5.0, 7.0, len(X_train)) + 0.001 * X_train.iloc[:, 0]
    model = Ridge(alpha=0.1).fit(X_train, y_train)
    bundle = build_qsar_prediction_bundle(
        model,
        feature_names=list(usable_cols),
        target_label="mordred_activity",
    )

    query = pd.DataFrame(
        {
            "compound_id": ["M1", "M2", "M3"],
            "SMILES": ["CCO", "CCCN", "c1ccncc1"],
        }
    )
    result = predict_with_qsar_model(
        bundle,
        query,
        QSARPredictionPackagerConfig(
            id_column="compound_id",
            target_label="mordred_activity",
        ),
    )

    assert len(result.predictions) == 3
    assert result.predictions["predicted_pActivity"].notna().all()
    assert result.package_manifest["recipe_kind"] == "mordred_selected"
    assert result.package_manifest["rows_failed"] == 0


def test_prediction_packager_manifest_includes_bundle_metadata():
    train_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCO", "CCCl"]
    X_train = _rdkit_training_frame(train_smiles)[["MolWt", "MolLogP"]]
    y_train = 0.02 * X_train["MolWt"] + 0.40 * X_train["MolLogP"]
    model = LinearRegression().fit(X_train, y_train)
    bundle = build_qsar_prediction_bundle(
        model,
        feature_names=list(X_train.columns),
        target_label="BoilingPoint",
        recipe_kind="rdkit_compact",
        model_name="Random Forest",
        source_widget="QSAR/QSPR Model Hub",
        training_rows=len(X_train),
        selected_feature_names=["MolWt"],
    )

    query = pd.DataFrame(
        {
            "compound_id": ["Q1", "Q2"],
            "canonical_smiles": ["CCO", "CCCN"],
        }
    )
    result = predict_with_qsar_model(
        bundle,
        query,
        QSARPredictionPackagerConfig(id_column="compound_id"),
    )

    assert result.package_manifest["bundle_model_name"] == "Random Forest"
    assert result.package_manifest["bundle_source_widget"] == "QSAR/QSPR Model Hub"
    assert result.package_manifest["bundle_training_rows"] == len(X_train)
    assert result.package_manifest["bundle_selected_feature_count"] == 1
    assert result.package_manifest["bundle_selected_feature_names"] == ["MolWt"]


def test_build_qsar_prediction_bundle_preserves_mlr_style_feature_contract():
    mlr_like = SimpleNamespace(
        x_names_after_preprocess=["MolWt", "MolLogP", "TPSA"],
        selected_names=["MolWt", "TPSA"],
        y_name="pActivity",
        predict=lambda X: np.asarray([5.0] * len(X), dtype=float),
    )

    bundle = build_qsar_prediction_bundle(
        mlr_like,
        target_label="pActivity",
        source_widget="MLR Model Selection",
        model_name="Multiple Linear Regression",
        selected_feature_names=["MolWt", "TPSA"],
    )

    assert isinstance(bundle, QSARPredictionModelBundle)
    assert list(bundle.feature_names) == ["MolWt", "MolLogP", "TPSA"]
    assert list(bundle.selected_feature_names) == ["MolWt", "TPSA"]
    assert bundle.source_widget == "MLR Model Selection"
    assert bundle.model_name == "Multiple Linear Regression"


def test_selected_feature_names_from_model_reads_selector_support():
    X = pd.DataFrame(
        {
            "MolWt": [100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
            "MolLogP": [1.1, 1.3, 1.7, 2.0, 2.4, 2.8],
            "TPSA": [20.0, 18.0, 16.0, 15.0, 12.0, 10.0],
        }
    )
    y = np.array([5.0, 5.4, 5.8, 6.0, 6.4, 6.8], dtype=float)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(f_regression, k=2)),
            ("model", LinearRegression()),
        ]
    )
    pipe.fit(X, y)

    selected = selected_feature_names_from_model(pipe, fallback_features=list(X.columns))

    assert len(selected) == 2
    assert set(selected).issubset(set(X.columns))


def test_write_model_bundle_package_writes_fair_artifacts(tmp_path):
    train_smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCO", "CCCl"]
    X_train = _rdkit_training_frame(train_smiles)[["MolWt", "MolLogP"]]
    y_train = 0.02 * X_train["MolWt"] + 0.40 * X_train["MolLogP"]
    model = LinearRegression().fit(X_train, y_train)
    bundle = build_qsar_prediction_bundle(
        model,
        feature_names=list(X_train.columns),
        target_label="BoilingPoint",
        recipe_kind="rdkit_compact",
        model_name="Linear Regression",
        source_widget="QSAR/QSPR Model Hub",
        training_rows=len(X_train),
        selected_feature_names=["MolWt"],
        training_summary={"test_r2": 0.82, "target_unit": "degC"},
    )

    paths = write_model_bundle_package(bundle, tmp_path / "boiling_point_model.pkl")
    loaded = load_model_pickle(paths["model_pickle"])
    manifest = pd.read_json(paths["manifest_json"], typ="series")

    assert Path(paths["model_pickle"]).exists()
    assert Path(paths["manifest_json"]).exists()
    assert Path(paths["feature_names_txt"]).exists()
    assert Path(paths["selected_features_txt"]).exists()
    assert isinstance(loaded, QSARPredictionModelBundle)
    assert loaded.target_label == "BoilingPoint"
    assert loaded.training_summary["target_unit"] == "degC"
    assert manifest["artifact_kind"] == "qsar_prediction_model_bundle"
    assert manifest["selected_feature_names"] == ["MolWt"]


class TestOWQSARPredictionPackager(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWQSARPredictionPackager)

    def test_model_input_autofills_dependent_variable_and_prediction_column(self):
        bundle = build_qsar_prediction_bundle(
            LinearRegression(),
            feature_names=["MolWt", "MolLogP"],
            target_label="BoilingPoint",
            recipe_kind="rdkit_compact",
        )

        self.widget.set_model(bundle)

        assert self.widget.target_label == "BoilingPoint"
        assert self.widget.prediction_column == "predicted_BoilingPoint"
        assert self.widget._edit_target_label.text() == "BoilingPoint"
        assert self.widget._edit_prediction_column.text() == "predicted_BoilingPoint"

    def test_molecules_input_is_normalized_to_prediction_frame(self):
        molecules = [
            ChemMol.from_smiles("CCO", name="ethanol"),
            "c1ccccc1",
        ]

        df = self.widget._molecules_to_df(molecules)

        assert list(df["canonical_smiles"]) == ["CCO", "c1ccccc1"]
        assert df.loc[0, "name"] == "ethanol"
        assert df.loc[0, "compound_id"]
        assert df.loc[1, "compound_id"] == "compound_0002"
