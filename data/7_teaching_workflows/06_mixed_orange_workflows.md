# Mixed Chemoinformatics + Orange Teaching Workflows

This note adds five more teaching examples that explicitly combine
`chem-inf-widgets` with classic Orange widgets such as projections,
clustering, heat maps, and outlier analysis.

These are meant as classroom-ready experiment ideas. Each one includes:

- a suggested dataset
- a recommended widget chain
- the learning goal
- a few questions students can investigate

## 1. Chemical Space Map With PCA Or t-SNE

Suggested dataset:
- `04_qsar_vegfr2_ic50.csv`

Widget chain:
- `File`
- `Mol Standardizer`
- `Fingerprint Generator`
- `PCA` or `t-SNE`
- `Scatter Plot`
- `Scaffold Analysis`
- `Molecular Viewer`

Goal:
- show how a molecular collection occupies chemical space
- compare scaffold families in a low-dimensional projection

What students should look for:
- whether compounds cluster by scaffold
- whether high-activity compounds occupy a specific region
- whether outlying projected points correspond to unusual chemistry

Suggested classroom prompts:
- Do the strongest compounds form one dense region or several?
- Are acyclic compounds or edge-case structures separated from the main cloud?
- Does PCA tell the same visual story as t-SNE?

## 2. Similarity Clustering And Cluster Inspection

Suggested dataset:
- `02_diversity_smiles.csv`

Widget chain:
- `File`
- `Mol Standardizer`
- `Fingerprint Generator`
- `Distances`
- `Hierarchical Clustering`
- `Heat Map`
- `Data Table`
- `Molecular Viewer`

Goal:
- teach how similarity matrices drive clustering
- connect cluster structure to actual scaffold or analog series membership

What students should look for:
- how many major chemical families are present
- whether clusters are scaffold-pure or chemically mixed
- whether heat map blocks correspond to intuitive analog series

Suggested classroom prompts:
- Where would you cut the dendrogram to define chemical families?
- Are there singleton molecules that look chemically unusual?
- How does this view compare with the `Diversity Picker` output?

## 3. Outlier Hunting In Descriptor Space

Suggested dataset:
- `04_qsar_vegfr2_ic50.csv`

Widget chain:
- `File`
- `Mol Standardizer`
- `Fingerprint Generator`
- `PCA`
- `Outliers`
- `Applicability Domain`
- `Scatter Plot`
- `Molecular Viewer`

Goal:
- distinguish statistical outliers from chemically meaningful novelty
- connect geometric outliers to model domain limits

What students should look for:
- which compounds are flagged as outliers by Orange
- whether the same compounds are also outside the applicability domain
- whether outliers belong to rare scaffolds or malformed chemistry

Suggested classroom prompts:
- Are all statistical outliers also chemically suspicious?
- Which compounds are only weakly outlying but still outside the AD?
- Should you remove these compounds before modeling or keep them?

## 4. Activity Landscape With Projections And Cliffs

Suggested dataset:
- `03_activity_cliffs_vegfr2_ic50.csv`

Widget chain:
- `File`
- `Mol Standardizer`
- `Fingerprint Generator`
- `t-SNE` or `PCA`
- `Scatter Plot`
- `Activity Cliff Finder`
- `Pair Viewer`
- `Molecular Viewer`

Goal:
- connect the global activity landscape with local cliff behavior
- show that nearby compounds in chemical space may still differ strongly in potency

What students should look for:
- regions with many high-potency compounds
- whether cliffs are concentrated inside one cluster or between nearby series
- whether the most informative SAR examples sit in dense or sparse neighborhoods

Suggested classroom prompts:
- Are activity cliffs isolated curiosities or repeated patterns?
- Do strong cliffs appear inside the same scaffold family?
- What structural change seems to drive the potency jump?

## 5. Cluster-Aware QSAR Interpretation

Suggested dataset:
- `04_qsar_vegfr2_ic50.csv`

Widget chain:
- `File`
- `Mol Standardizer`
- `Fingerprint Generator`
- `k-Means` or `Hierarchical Clustering`
- `Box Plot`
- `QSAR Regression`
- `Scatter Plot`
- `Molecular Viewer`

Goal:
- compare activity distributions between chemically defined clusters
- test whether model errors concentrate in specific regions of chemical space

What students should look for:
- whether some clusters are consistently more active than others
- whether one cluster is much harder for the model to predict
- whether selected mispredicted compounds belong to a single analog series

Suggested classroom prompts:
- Is the model equally good across all clusters?
- Which cluster has the widest activity spread?
- Would scaffold-based splitting be fairer than a random split here?

## Recommended Reuse Of Existing Teaching Data

If you want a compact teaching module, these pairings work well:

- `02_diversity_smiles.csv`
  best for clustering and similarity-family exercises
- `03_activity_cliffs_vegfr2_ic50.csv`
  best for cliff and local SAR interpretation
- `04_qsar_vegfr2_ic50.csv`
  best for projections, outliers, clustering, and QSAR diagnostics

## Suggested Teaching Progression

One nice sequence for a lab session is:

1. Start with `Chemical Space Map`
2. Continue to `Similarity Clustering`
3. Move into `Activity Landscape With Cliffs`
4. Build `QSAR Regression`
5. Finish with `Outlier Hunting` and `Applicability Domain`

This sequence lets students move from:

- raw structural diversity
- to chemical families
- to SAR interpretation
- to predictive modeling
- to model trustworthiness
