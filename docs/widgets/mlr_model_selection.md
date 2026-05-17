# MLR Model Selection

## Status

Current widget in the package.

Source:
- [ow_mlr_model_selection.py](../../src/chem_inf_widgets/widgets/ow_mlr_model_selection.py)

## Purpose

`MLR Model Selection` builds Multiple Linear Regression models with descriptor filtering and automated variable selection strategies. It produces QSAR-style diagnostic outputs including coefficient tables and an HTML report.

## Input

- `Data` — training data as Orange `Table` with numeric descriptor columns and a target variable
- `Test Data` — optional holdout set for external validation

## Output

- `Model` — fitted regression model
- `Train Results` — predicted vs. actual values on training data
- `Test Results` — predicted vs. actual values on test data
- `Predictions` — alias for test/holdout results
- `Coefficients` — term, coefficient, SE, t-statistic, p-value, VIF table
- `Report HTML` — full HTML diagnostic report with embedded plots

## Variable selection strategies

- Forward selection
- Backward elimination
- Monte Carlo selection
- Genetic Algorithm selection

## Typical workflow

1. `Mol Standardizer`
2. `Mol Descriptors` or `Fingerprint Generator`
3. `MLR Model Selection`
4. `Applicability Domain`

## Notes

- `MLR Model Selection` is complementary to `QSAR Regression`: use this widget when interpretability and variable selection matter; use `QSAR Regression` for ensemble/non-linear models.
- The HTML report is suitable for direct export to a teaching or research notebook.
