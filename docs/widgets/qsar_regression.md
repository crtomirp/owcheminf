# QSAR Regression

## Status

Current widget in the package.

Source:
- [ow_qsar_regression.py](../../src/chem_inf_widgets/widgets/ow_qsar_regression.py)
- [qsar_regression_service.py](../../src/chem_inf_widgets/chemcore/services/qsar_regression_service.py)

## Purpose

`QSAR Regression` builds and diagnoses regression models for structure-property and structure-activity tasks inside Orange.

It combines model fitting, basic preprocessing, hyperparameter tuning and diagnostic plots in a single widget.

## Input

- `Data`
- optional `External Data`

The input should already contain numeric descriptor columns and a numeric target variable.

## Output

- `Model`
- `Train Results`
- `Test Results`
- `External Results`
- `Selected Compounds`

## Supported models

Depending on installed extras, the widget supports combinations of:

- Random Forest
- Support Vector Regression
- Gradient Boosting
- PLS Regression
- Decision Tree Regression
- Lasso
- Ridge
- Elastic Net
- optional Deep Learning Regression when `torch` is available

## Main capabilities

- normalization
- imputation
- train/test split
- optional external evaluation set
- grid search or randomized search
- optional feature selection
- interactive diagnostics

## Interactive diagnostics

The diagnostic plots support point selection through rectangle or lasso tools. Selected points are sent out as `Selected Compounds`, which makes it easy to inspect outliers in downstream widgets.

## Typical workflow

1. `Mol Standardizer`
2. `Mol Descriptor`, `PaDEL Descriptors` or `Fingerprint Generator`
3. `QSAR Regression`
4. `Applicability Domain`
5. `Molecular Viewer` or `Compound Detail Card`

## Notes

- The widget becomes much more informative when paired with `Applicability Domain`.
- For fairer evaluation, consider preparing scaffold-aware splits upstream.
