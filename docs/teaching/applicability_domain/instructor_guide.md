# Instructor Guide — Applicability Domain

## Target audience

This module is suitable for undergraduate or master's level courses in cheminformatics, medicinal chemistry, computational chemistry, pharmaceutical sciences, environmental chemistry, or data-driven chemistry.

## Recommended duration

- Short version: 45–60 minutes
- Full practical: 2–3 hours
- Capstone extension: 1–2 weeks

## Prior knowledge

Students should know:

- basic molecular descriptors,
- training/test split concept,
- basic QSAR/QSPR modeling,
- scatter plots and outliers,
- simple interpretation of descriptor tables.

## Suggested teaching sequence

1. Start with the conceptual question: *Should a QSAR model predict any molecule in the universe?*
2. Show a normal query molecule and an extreme query molecule.
3. Run the Applicability Domain widget.
4. Compare `AD_in_domain=True` and `AD_in_domain=False` compounds.
5. Discuss why a prediction may be numerically produced but chemically unreliable.
6. Connect this with OECD QSAR validation principles and responsible reporting.

## Main concept map

```text
Training/reference compounds
        ↓
Descriptor space
        ↓
AD boundary / distance threshold
        ↓
New query compound
        ↓
Inside domain? → prediction more defensible
Outside domain? → prediction should be flagged
```

## Recommended classroom questions

1. Can a model make a prediction outside its training domain?
2. Should we trust that prediction?
3. What is the difference between interpolation and extrapolation?
4. Can a model have good cross-validation performance but poor reliability for a new molecule?
5. Why can descriptor choice change the applicability domain?

## Orange setup

Use two File widgets:

```text
File 1: examples/applicability_domain/ad_reference_training_set.csv
File 2: examples/applicability_domain/ad_query_prediction_set.csv
```

Connect:

```text
File 1 → Applicability Domain (Reference Data)
File 2 → Applicability Domain (Data)
Applicability Domain → Data Results → Data Table
Applicability Domain → AD Summary → Data Table
```

## Interpreting outputs

### Data Results

Contains the original query data plus added AD columns:

- `AD_leverage`
- `AD_in_williams`
- `AD_knn_dist`
- `AD_in_knn`
- `AD_maha_d2`, if enabled
- `AD_in_maha`, if enabled
- `AD_in_domain`

### Reference Results

Contains the reference set scored against its own domain. This is useful for identifying unusual training compounds.

### AD Summary

Reports:

- number of reference rows,
- number of query rows,
- number of shared features,
- Williams threshold `h*`,
- number of compounds inside the domain,
- kNN/Mahalanobis thresholds if enabled.

## Assessment idea

Give students three compounds:

1. one clearly inside domain,
2. one borderline,
3. one outside domain.

Ask them to justify whether each prediction should be accepted, flagged, or rejected.

## Common misconceptions

### Misconception 1: Outside AD means the molecule is inactive.

Correction: Outside AD means the model has insufficient support for reliable prediction. It does not directly imply inactivity.

### Misconception 2: Inside AD guarantees a correct prediction.

Correction: Inside AD only improves confidence that the model is being applied in a supported region. Model error still exists.

### Misconception 3: AD is independent of descriptors.

Correction: AD is defined in descriptor space. Changing descriptors changes the geometry of the domain.

## Suggested homework

Students prepare a short QSAR report section titled **Applicability Domain and Prediction Reliability** using the output of the widget.
