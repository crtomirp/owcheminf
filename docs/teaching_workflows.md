# Teaching Workflows

The repository contains a curated set of classroom-ready datasets and Orange workflows for chemoinformatics teaching.

## Where to find them

- datasets: [data/7_teaching_workflows](../data/7_teaching_workflows)
- Orange workflows: [data/ows_workflow](../data/ows_workflow)
- workflow index: [data/ows_workflow/README.md](../data/ows_workflow/README.md)

## Core workflow themes

### Data curation

- standardization
- filtering
- scaffold inspection

### Representation and diversity

- fingerprints
- diversity picking
- chemical space maps
- clustering

### SAR and medicinal chemistry

- activity cliffs
- pair viewing
- compound-detail driven search
- motif-based search

### Modeling

- QSAR regression
- applicability domain
- cluster-aware interpretation

## Suggested classroom sequence

1. `01_standardization_and_filtering.ows`
2. `02_diversity_picker.ows`
3. `03_activity_cliffs.ows`
4. `04_qsar_regression.ows`
5. `05_applicability_domain.ows`

Then continue with mixed Orange workflows such as:

- `06_chemical_space_map.ows`
- `07_similarity_clustering.ows`
- `08_outlier_hunting.ows`
- `09_activity_landscape.ows`
- `10_cluster_aware_qsar.ows`

And the SmartChemist-style search demos:

- `11_compound_detail_database_search.ows`
- `12_motif_query_substructure_search.ows`
- `13_multi_motif_hybrid_search.ows`

## Practical teaching note

The `.ows` files intentionally do not hardcode local paths, so after opening them you should load the matching CSV file in the `File` widget manually.
