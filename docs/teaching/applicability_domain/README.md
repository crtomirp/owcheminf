# Applicability Domain Teaching Module

This module provides teaching material for the **Applicability Domain** widget in the Chem-Inf Orange add-on.

Widget location:

```text
Cheminf - Processing → Applicability Domain
```

## Why teach Applicability Domain?

A QSAR/QSPR model should not be treated as reliable for every possible molecule. The **applicability domain (AD)** describes the chemical or descriptor space where the model has enough support from the training/reference data. Predictions outside this domain should be interpreted cautiously.

This teaching module focuses on three AD ideas supported by the widget:

1. **Williams leverage** — identifies compounds with unusual descriptor combinations relative to the reference set.
2. **kNN distance** — identifies compounds far from nearby reference molecules in standardized descriptor space.
3. **Mahalanobis distance** — identifies multivariate distance from the reference distribution.

## Learning outcomes

After completing this module, students should be able to:

1. Explain why QSAR predictions require an applicability domain.
2. Distinguish between model performance and prediction reliability.
3. Use the Applicability Domain widget with reference and query datasets.
4. Interpret `AD_leverage`, `AD_in_williams`, `AD_knn_dist`, `AD_in_knn`, `AD_maha_d2`, `AD_in_maha`, and `AD_in_domain`.
5. Identify descriptor-space outliers and discuss whether their predictions should be trusted.
6. Connect AD analysis with external validation and responsible QSAR reporting.

## Files in this module

```text
README.md
instructor_guide.md
worksheet_01_conceptual_intro.md
worksheet_02_williams_leverage.md
worksheet_03_knn_distance_domain.md
worksheet_04_mahalanobis_domain.md
worksheet_05_external_prediction_reliability.md
worksheet_06_descriptor_choice_effect.md
worksheet_07_fingerprint_domain_discussion.md
worksheet_08_qsar_reporting_with_ad.md
worksheet_09_case_study_outlier_triage.md
worksheet_10_capstone_ad_qsar_project.md
```

## Example data

```text
examples/applicability_domain/ad_reference_training_set.csv
examples/applicability_domain/ad_query_prediction_set.csv
```

These files include simplified descriptor columns for teaching. In real projects, descriptors should be calculated from structures using descriptor widgets.

## Basic Orange workflow

```text
File(reference training set) ─────────────┐
                                          ↓ Reference Data
File(query/external set) → Applicability Domain → Data Results → Data Table
                                          ↓
                                       AD Summary → Data Table
```

## Recommended settings for the first lesson

- Williams leverage: ON
- kNN distance: ON
- kNN k: 5
- kNN quantile: 0.95
- Mahalanobis distance: OFF at first
- Combine: `and`

Then repeat with Mahalanobis enabled and compare the classification of borderline compounds.

## Important teaching warning

Applicability domain is not a guarantee that a prediction is correct. It is a warning system that asks: **is this molecule similar enough to the reference data that the model has a reasonable basis for prediction?**
