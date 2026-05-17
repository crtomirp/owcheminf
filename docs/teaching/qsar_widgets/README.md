# QSAR Widgets Teaching Materials

This folder contains English teaching materials for the QSAR-related widgets in `chem-inf-widgets`.

## Covered widgets

| Widget | Main teaching role |
|---|---|
| Fingerprint Generator | Morgan/RDKit/MACCS fingerprints. |
| Cyclic Registry Fingerprint | Interpretable 4096-bit fingerprint with matched registry explanations. |
| Mol Descriptors 2 | Molecular descriptors from SMILES or molecule objects. |
| PaDEL Descriptors | PaDEL descriptors when Java/PaDEL is available. |
| ISIDA Descriptors | Fragment-count descriptors. |
| Scaffold Splitter | Scaffold-aware train/validation/test splits. |
| MLR Model Selection | Interpretable multiple linear regression with descriptor selection. |
| QSAR Regression | Flexible QSAR regression and external prediction. |
| Activity Cliff Finder | Similar compound pairs with large property/activity differences. |

## Recommended order

1. Descriptor and fingerprint preparation.
2. Simple QSAR regression.
3. Scaffold-aware validation.
4. Interpretable MLR.
5. Model comparison.
6. External prediction.
7. Activity cliffs.
8. Reproducible reporting.
9. Capstone QSAR project.

## Example data

Use:

```text
examples/qsar_widgets/qsar_training_set.csv
examples/qsar_widgets/qsar_external_prediction_set.csv
```

The numeric target values are simplified educational values and must not be treated as experimental assay data.

## Related teaching module

For model reliability and external prediction interpretation, see:

```text
docs/teaching/applicability_domain/
```

This module teaches Williams leverage, kNN distance, Mahalanobis distance, and QSAR reporting with applicability-domain flags.
