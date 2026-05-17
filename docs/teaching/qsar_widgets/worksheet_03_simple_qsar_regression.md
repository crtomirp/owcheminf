# Worksheet 03: Building a Simple QSAR Regression Model

**Estimated time:** 90 min  
**Level:** intermediate  
**Main widgets:** Mol Descriptors 2, Fingerprint Generator, QSAR Regression, Data Table

## Context

A QSAR model relates molecular structure to a numerical property such as activity, solubility, toxicity, or docking score.

## Intended learning outcomes

Students will be able to:

1. choose a target variable,
2. train a regression model,
3. inspect train/test predictions,
4. explain the difference between fitting and prediction,
5. identify signs of overfitting.

## Input data

Use:

```text
examples/qsar_widgets/qsar_training_set.csv
```

Target column:

```text
activity_pIC50_educational
```

## Orange workflow

```text
File → Mol Descriptors 2 → QSAR Regression → Data Table
```

Alternative:

```text
File → Fingerprint Generator → QSAR Regression → Data Table
```

## Student tasks

1. Load the dataset.
2. Calculate descriptors or fingerprints.
3. Set `activity_pIC50_educational` as the target variable.
4. Train a regression model.
5. Inspect train and test outputs.
6. Record the available model diagnostics.

## Guiding questions

- Why is the target variable not allowed to be used as an input descriptor?
- What does a residual mean?
- What does it mean if training performance is much better than test performance?
- Why is this dataset too small for scientific conclusions?

## Expected output

Train/test prediction tables with observed and predicted values.
