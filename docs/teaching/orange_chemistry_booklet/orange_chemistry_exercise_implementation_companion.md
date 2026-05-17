# Orange in Chemistry: Exercise and Implementation Companion

This companion consolidates all classroom exercises and implementation instructions for the Orange-based chemistry teaching packages.

## How to use this companion

1. Start with one 45-90 minute worksheet. 2. Use the listed example data. 3. Ask students to save both the Orange workflow and a short report. 4. End with reflection on chemistry, data quality, and reproducibility.


## Module: Cyclic Registry Fingerprint

**Big idea:** Use an interpretable 4096-bit fingerprint to connect molecular structure, cyclic motifs, and machine-learning-ready representations.

**Core workflow:** `File -> Cyclic Registry Fingerprint -> Data Table / Matched Registry Entries`

**Suggested data:** `examples/cyclic_registry_fingerprint/cyclic_registry_training_set.csv`


### Exercise 1: Drug Heterocycles
- Objective: Identify heterocycles in drug-like molecules and explain why matched registry entries matter.
- Orange workflow: `File -> Cyclic Registry Fingerprint -> Matched Registry Entries -> Data Table`
- Implementation: Create a frequency table of detected heterocycle entries and annotate three molecules.
- Assessment evidence: Students should explain at least two detected ring systems using chemical language.


### Exercise 2: Aromatic vs Kekule
- Objective: Compare aromatic and kekulized encodings of the same ring families.
- Orange workflow: `File -> Cyclic Registry Fingerprint -> Matched Registry Entries`
- Implementation: Run paired SMILES examples and compare registry matches.
- Assessment evidence: Students should recognize representation robustness and remaining edge cases.


### Exercise 3: Natural Products
- Objective: Profile cyclic motifs in natural-product-like structures.
- Orange workflow: `File -> Cyclic Registry Fingerprint -> Data Table`
- Implementation: Group matched entries by family and discuss natural product diversity.
- Assessment evidence: Students should connect motifs to compound class.


### Exercise 4: QSAR/QSPR Input
- Objective: Use the 4096-bit output as QSAR features.
- Orange workflow: `File -> Cyclic Registry Fingerprint -> Test & Score`
- Implementation: Compare a simple model using registry-enhanced features.
- Assessment evidence: Students should separate prediction from interpretation.


### Exercise 5: Feature Interpretation
- Objective: Relate high-ranking model features to matched registry entries.
- Orange workflow: `Cyclic Registry Fingerprint -> Rank + Matched Registry Entries`
- Implementation: Find top-ranked bits and inspect which motifs trigger them.
- Assessment evidence: Students should state whether a bit is interpretable or hashed.


### Exercise 6: Library Comparison
- Objective: Compare motif profiles between libraries.
- Orange workflow: `File -> Cyclic Registry Fingerprint -> Matched Registry Entries`
- Implementation: Compare frequency profiles for two groups in the input table.
- Assessment evidence: Students should present one structural difference between libraries.


### Exercise 7: Docking Results
- Objective: Describe motifs among top-ranked docking candidates.
- Orange workflow: `File -> Select Rows -> Cyclic Registry Fingerprint`
- Implementation: Filter top docking scores and profile cyclic motifs.
- Assessment evidence: Students should avoid claiming causality from docking score alone.


### Exercise 8: Unwanted Motifs
- Objective: Screen for undesirable functional or structural motifs.
- Orange workflow: `Cyclic Registry Fingerprint -> Matched Registry Entries -> Select Rows`
- Implementation: Filter matched entries by family/section and flag compounds.
- Assessment evidence: Students should distinguish alerting from automatic rejection.


### Exercise 9: SMARTS Learning
- Objective: Use registry hits to learn SMARTS/substructure thinking.
- Orange workflow: `File -> Cyclic Registry Fingerprint -> Matched Registry Entries`
- Implementation: Predict hits before running the widget, then compare results.
- Assessment evidence: Students should explain false expectations constructively.


### Exercise 10: Method Development
- Objective: Use validation/collision reporting to improve an interpretable fingerprint.
- Orange workflow: `CLI validator + Cyclic Registry Fingerprint`
- Implementation: Run registry validation and discuss collisions.
- Assessment evidence: Students should propose one improvement to registry metadata.


## Module: QSAR Widgets

**Big idea:** Build, validate, compare, and report QSAR models while keeping chemistry and data leakage in view.

**Core workflow:** `File -> Mol Descriptors 2 / Fingerprint Generator -> QSAR Regression -> Test & Score`

**Suggested data:** `examples/qsar_widgets/qsar_training_set.csv`


### Exercise 1: Descriptor Preparation
- Objective: Generate molecular descriptors and inspect missing or constant features.
- Orange workflow: `File -> Mol Descriptors 2 -> Data Table`
- Implementation: Identify descriptors with missing values and discuss preprocessing.
- Assessment evidence: Students should document descriptor settings.


### Exercise 2: Fingerprint QSAR
- Objective: Use binary fingerprints as model features.
- Orange workflow: `File -> Fingerprint Generator -> QSAR Regression`
- Implementation: Compare Morgan and cyclic registry fingerprints.
- Assessment evidence: Students should explain fingerprint radius/bit length tradeoffs.


### Exercise 3: Simple QSAR Regression
- Objective: Train a baseline regression model for a continuous endpoint.
- Orange workflow: `File -> Mol Descriptors 2 -> QSAR Regression -> Predictions`
- Implementation: Fit a model and report RMSE/MAE/R2.
- Assessment evidence: Students should interpret metrics cautiously.


### Exercise 4: Scaffold Split Validation
- Objective: Test whether performance survives scaffold-aware splitting.
- Orange workflow: `Mol Descriptors 2 -> Scaffold Splitter -> QSAR Regression`
- Implementation: Compare random and scaffold split performance.
- Assessment evidence: Students should identify overoptimistic random-split results.


### Exercise 5: Interpretable MLR
- Objective: Build an interpretable multiple-linear-regression model.
- Orange workflow: `Mol Descriptors 2 -> MLR Model Selection`
- Implementation: Select few descriptors and discuss coefficients.
- Assessment evidence: Students should avoid overfitting and unsupported mechanistic claims.


### Exercise 6: Model Comparison
- Objective: Compare descriptor, Morgan, and cyclic-registry feature sets.
- Orange workflow: `Three feature branches -> Test & Score`
- Implementation: Create a model comparison table.
- Assessment evidence: Students should report both performance and interpretability.


### Exercise 7: External Prediction
- Objective: Apply the model to unseen molecules.
- Orange workflow: `Training branch + external File -> Predictions`
- Implementation: Predict an external set and flag questionable cases.
- Assessment evidence: Students should connect prediction to applicability domain.


### Exercise 8: Activity Cliffs
- Objective: Investigate why similar molecules can have different activity.
- Orange workflow: `Fingerprint Generator -> Activity Cliff Finder`
- Implementation: Identify cliff pairs and inspect structures.
- Assessment evidence: Students should describe SAR implications.


### Exercise 9: QSAR Reporting
- Objective: Prepare a reproducible QSAR report.
- Orange workflow: `Full QSAR workflow -> Save/Report`
- Implementation: Document data, preprocessing, split, model, metrics, limitations.
- Assessment evidence: Students should include provenance and limitations.


### Exercise 10: Capstone QSAR Project
- Objective: Design a small end-to-end QSAR study.
- Orange workflow: `Standardizer -> Features -> Split -> Model -> AD -> Report`
- Implementation: Complete a concise project report.
- Assessment evidence: Students should justify every workflow choice.


## Module: Applicability Domain

**Big idea:** Decide when a QSAR prediction should be trusted, questioned, or rejected.

**Core workflow:** `Reference File + Query File -> Applicability Domain -> Data Results / AD Summary`

**Suggested data:** `examples/applicability_domain/ad_reference_training_set.csv + ad_query_prediction_set.csv`


### Exercise 1: Conceptual Introduction
- Objective: Explain why a model has a domain of reliable use.
- Orange workflow: `Training/query data -> Applicability Domain`
- Implementation: Classify molecules as inside/outside domain.
- Assessment evidence: Students should distinguish model error from domain warning.


### Exercise 2: Williams Leverage
- Objective: Use leverage to identify structurally influential query compounds.
- Orange workflow: `Descriptors -> Applicability Domain`
- Implementation: Interpret leverage and warning thresholds.
- Assessment evidence: Students should explain the Williams plot concept.


### Exercise 3: kNN Distance Domain
- Objective: Use nearest-neighbour distances to assess local similarity.
- Orange workflow: `Descriptors/Fingerprints -> Applicability Domain`
- Implementation: Compare k values and distance thresholds.
- Assessment evidence: Students should relate distance to chemical similarity.


### Exercise 4: Mahalanobis Domain
- Objective: Use multivariate distance to flag unusual compounds.
- Orange workflow: `Descriptor table -> Applicability Domain`
- Implementation: Run Mahalanobis-based domain assessment.
- Assessment evidence: Students should discuss scaling and descriptor correlation.


### Exercise 5: External Prediction Reliability
- Objective: Combine predictions with AD flags.
- Orange workflow: `QSAR Predictions -> Applicability Domain -> Data Table`
- Implementation: Mark predictions as reliable/caution/outside domain.
- Assessment evidence: Students should report uncertainty labels.


### Exercise 6: Descriptor Choice Effect
- Objective: Show that AD depends on representation.
- Orange workflow: `Two descriptor/fingerprint branches -> AD`
- Implementation: Compare AD flags across feature spaces.
- Assessment evidence: Students should not treat AD as absolute truth.


### Exercise 7: Fingerprint Domain Discussion
- Objective: Discuss AD in binary fingerprint space.
- Orange workflow: `Fingerprint Generator -> Applicability Domain`
- Implementation: Compare Tanimoto-like thinking with descriptor distance.
- Assessment evidence: Students should identify limitations.


### Exercise 8: QSAR Reporting with AD
- Objective: Add AD results to model reporting.
- Orange workflow: `Full QSAR + AD workflow`
- Implementation: Prepare a report paragraph for AD.
- Assessment evidence: Students should include rule, threshold, and consequence.


### Exercise 9: Outlier Triage
- Objective: Investigate why compounds are outside domain.
- Orange workflow: `AD Results -> Data Table -> Viewer`
- Implementation: Inspect flagged molecules and propose actions.
- Assessment evidence: Students should recommend remove, retrain, or caution.


### Exercise 10: Capstone AD-QSAR
- Objective: Build a QSAR model with an explicit reliability layer.
- Orange workflow: `Standardizer -> QSAR -> AD -> Report`
- Implementation: Deliver a prediction table with AD decisions.
- Assessment evidence: Students should justify the decision policy.


## Module: Molecular Standardization

**Big idea:** Clean molecular structures before descriptors, fingerprints, modelling, or interpretation.

**Core workflow:** `File -> Mol Standardizer -> Data Table / downstream feature widget`

**Suggested data:** `examples/molecular_standardization/standardization_training_set.csv`


### Exercise 1: Why Standardization Matters
- Objective: Show that representation choices affect downstream results.
- Orange workflow: `File -> Mol Standardizer -> Data Table`
- Implementation: Compare original and standardized SMILES.
- Assessment evidence: Students should identify at least three changed records.


### Exercise 2: Invalid SMILES and Sanitization
- Objective: Detect and document invalid structures.
- Orange workflow: `File -> Mol Standardizer -> Data Table`
- Implementation: Inspect failed rows and error/status columns.
- Assessment evidence: Students should not silently discard failures.


### Exercise 3: Salts and Mixtures
- Objective: Handle salts, mixtures, and largest-fragment choices.
- Orange workflow: `File -> Mol Standardizer`
- Implementation: Compare preserve-salts and largest-fragment logic.
- Assessment evidence: Students should explain when each choice is appropriate.


### Exercise 4: Charge Normalization
- Objective: Explore uncharging/reionization effects.
- Orange workflow: `Mol Standardizer -> Data Table`
- Implementation: Track changes in charged examples.
- Assessment evidence: Students should preserve chemically meaningful charges when needed.


### Exercise 5: Nitro/Zwitterions/Quaternary Ammonium
- Objective: Test common valence-sensitive cases.
- Orange workflow: `Mol Standardizer -> Data Table`
- Implementation: Check nitro and quaternary ammonium examples.
- Assessment evidence: Students should understand why formal charges matter.


### Exercise 6: Aromaticity and Kekulization
- Objective: Compare aromatic and kekulized representations.
- Orange workflow: `Mol Standardizer -> Cyclic Registry Fingerprint`
- Implementation: Evaluate downstream motif matching.
- Assessment evidence: Students should document representation effects.


### Exercise 7: Tautomers and Protonation States
- Objective: Discuss what standardization does and does not solve.
- Orange workflow: `Mol Standardizer -> Data Table`
- Implementation: Compare related protonation/tautomer examples.
- Assessment evidence: Students should avoid pretending pH was fully modelled.


### Exercise 8: Before Fingerprints
- Objective: Show standardization effect on fingerprint similarity.
- Orange workflow: `Mol Standardizer -> Fingerprint Generator`
- Implementation: Compare bit outputs before/after cleaning.
- Assessment evidence: Students should report how preprocessing changes features.


### Exercise 9: Before QSAR
- Objective: Show standardization as part of modelling workflow.
- Orange workflow: `Mol Standardizer -> Descriptors -> QSAR`
- Implementation: Train a small model with clean data.
- Assessment evidence: Students should describe preprocessing in methods.


### Exercise 10: Reproducible Standardization Report
- Objective: Create a standardization audit trail.
- Orange workflow: `Mol Standardizer -> Data Table/Report`
- Implementation: Write a protocol with settings and failures.
- Assessment evidence: Students should include profile name and changed records.


## Module: ChEMBL Bioactivity Curation

**Big idea:** Transform heterogeneous public bioactivity records into a transparent QSAR-ready dataset.

**Core workflow:** `ChEMBL Browser -> Bioactivity Retriever -> Standardizer -> QSAR features`

**Suggested data:** `examples/chembl_bioactivity_curation/chembl_bioactivity_curation_demo.csv`


### Exercise 1: What is ChEMBL Data
- Objective: Identify entities: compound, target, assay, document, activity.
- Orange workflow: `Demo CSV -> Data Table`
- Implementation: Label key columns and explain relationships.
- Assessment evidence: Students should not treat all rows as equivalent measurements.


### Exercise 2: Target Search and Confidence
- Objective: Use target confidence to select biologically meaningful records.
- Orange workflow: `ChEMBL Browser -> Data Table`
- Implementation: Compare targets with different confidence scores.
- Assessment evidence: Students should justify target selection.


### Exercise 3: Endpoints/Units/Relations
- Objective: Curate IC50/Ki/Kd/EC50 and relation symbols.
- Orange workflow: `Bioactivity table -> Select Rows`
- Implementation: Filter consistent endpoint and unit combinations.
- Assessment evidence: Students should explain pChEMBL and relation qualifiers.


### Exercise 4: Retrieving Bioactivities
- Objective: Download or inspect bioactivities for a target.
- Orange workflow: `ChEMBL Browser -> Bioactivity Retriever`
- Implementation: Create an initial raw table.
- Assessment evidence: Students should preserve raw data provenance.


### Exercise 5: Assay Filtering
- Objective: Filter by assay type, confidence, organism, and validity comments.
- Orange workflow: `Bioactivity Retriever -> Select Rows`
- Implementation: Create a curated subset.
- Assessment evidence: Students should document inclusion/exclusion rules.


### Exercise 6: Structure Cleaning
- Objective: Standardize retrieved compounds.
- Orange workflow: `Bioactivity table -> Mol Standardizer`
- Implementation: Inspect failures and salts.
- Assessment evidence: Students should connect curation and chemistry.


### Exercise 7: Duplicates and Aggregation
- Objective: Handle repeated measurements for the same compound.
- Orange workflow: `Data Table -> Group/Aggregate workflow`
- Implementation: Compare mean/median/best-case aggregation.
- Assessment evidence: Students should state aggregation rule.


### Exercise 8: QSAR-Ready Dataset
- Objective: Build a clean table for modelling.
- Orange workflow: `Curated Bioactivity -> Descriptors/Fingerprints`
- Implementation: Create final table with endpoint and features.
- Assessment evidence: Students should include units and transformed endpoint.


### Exercise 9: Provenance and FAIR Reporting
- Objective: Write a reproducible data curation protocol.
- Orange workflow: `Workflow plus exported tables`
- Implementation: Prepare a methods-ready curation paragraph.
- Assessment evidence: Students should report ChEMBL version/date if available.


### Exercise 10: Capstone Curation Project
- Objective: Prepare a small curated dataset from a target.
- Orange workflow: `ChEMBL -> Standardizer -> QSAR-ready table`
- Implementation: Deliver raw, curated, and modelling-ready files.
- Assessment evidence: Students should justify every filter.


## Module: Scaffold and Activity Cliff Analysis

**Big idea:** Use scaffolds and cliffs to connect chemical series, validation design, and medicinal chemistry interpretation.

**Core workflow:** `File -> Scaffold Analysis / Scaffold Splitter / Activity Cliff Finder`

**Suggested data:** `examples/scaffold_activity_cliff_analysis/scaffold_activity_training_set.csv`


### Exercise 1: What Scaffolds Represent
- Objective: Explain scaffold as a series-level abstraction.
- Orange workflow: `File -> Scaffold Analysis -> Data Table`
- Implementation: Compare molecule and scaffold columns.
- Assessment evidence: Students should avoid overinterpreting scaffold as full chemistry.


### Exercise 2: Murcko and Generic Scaffolds
- Objective: Compare Murcko and generic scaffold definitions.
- Orange workflow: `Scaffold Analysis -> Scaffold Summary`
- Implementation: Inspect scaffold grouping differences.
- Assessment evidence: Students should describe what atoms/bonds are abstracted.


### Exercise 3: Scaffold Frequency
- Objective: Profile dominant chemical series.
- Orange workflow: `Scaffold Analysis -> Summary`
- Implementation: Rank scaffolds by compound count.
- Assessment evidence: Students should identify series imbalance.


### Exercise 4: Scaffold Splitter
- Objective: Create scaffold-aware train/test splits.
- Orange workflow: `Scaffold Splitter -> Train/Test Data`
- Implementation: Inspect scaffold distribution across splits.
- Assessment evidence: Students should avoid scaffold leakage.


### Exercise 5: Random vs Scaffold Split
- Objective: Compare model performance under split strategies.
- Orange workflow: `QSAR workflow with random vs scaffold split`
- Implementation: Report performance differences.
- Assessment evidence: Students should explain why scaffold split is often harder.


### Exercise 6: Activity Cliff Concepts
- Objective: Define similar structure/different activity.
- Orange workflow: `File -> Fingerprint Generator -> Activity Cliff Finder`
- Implementation: Find example cliff pairs.
- Assessment evidence: Students should connect cliffs to SAR uncertainty.


### Exercise 7: Detecting Cliffs with Fingerprints
- Objective: Tune similarity and potency thresholds.
- Orange workflow: `Activity Cliff Finder`
- Implementation: Explore threshold sensitivity.
- Assessment evidence: Students should report parameter choices.


### Exercise 8: Medicinal Chemistry Interpretation
- Objective: Interpret transformations in cliff pairs.
- Orange workflow: `Activity Cliff Finder -> Data Table -> Viewer`
- Implementation: Discuss substituent changes and activity shifts.
- Assessment evidence: Students should avoid unsupported mechanism claims.


### Exercise 9: Scaffold-Cliff-QSAR-AD Link
- Objective: Connect cliffs to validation and domain warnings.
- Orange workflow: `Scaffold + Cliff + QSAR + AD workflow`
- Implementation: Identify difficult regions of chemical space.
- Assessment evidence: Students should synthesize multiple outputs.


### Exercise 10: Capstone Scaffold/Cliff Project
- Objective: Analyze a chemical series and report cliffs.
- Orange workflow: `Full scaffold/cliff workflow`
- Implementation: Deliver scaffold summary and selected cliff examples.
- Assessment evidence: Students should propose next design decisions.


## Module: Supplementary Cheminformatics Workflows

**Big idea:** Use core cheminformatics widgets for similarity, substructure search, diversity, library design, visualization, MMPA, reactions, and QC.

**Core workflow:** `Similarity Search / Substructure Search / Diversity Picker / MMP / Reaction Enumerator / Viewers`

**Suggested data:** `examples/supplementary_cheminformatics_workflows/*.csv`


### Exercise 1: Similarity Search
- Objective: Find molecules similar to a query and interpret Tanimoto scores.
- Orange workflow: `File -> Fingerprint Generator -> Similarity Search`
- Implementation: Rank library molecules by similarity.
- Assessment evidence: Students should state fingerprint and threshold.


### Exercise 2: Substructure SMARTS Search
- Objective: Search a library for a structural motif.
- Orange workflow: `File -> Substructure Search -> Data Table`
- Implementation: Use SMARTS examples and inspect matches.
- Assessment evidence: Students should explain SMARTS specificity.


### Exercise 3: Diversity Picker
- Objective: Select a small diverse subset from a larger library.
- Orange workflow: `File -> Fingerprint Generator -> Diversity Picker`
- Implementation: Pick a representative set and justify size.
- Assessment evidence: Students should distinguish diversity from quality.


### Exercise 4: Drug Filter/Likeness
- Objective: Apply drug-like or rule-based filters.
- Orange workflow: `File -> Drug Filter -> Data Table`
- Implementation: Compare accepted/rejected compounds.
- Assessment evidence: Students should treat filters as heuristics.


### Exercise 5: SDF I/O and Provenance
- Objective: Read/write molecule files while keeping metadata.
- Orange workflow: `SDF Reader -> Processing -> SDF Writer`
- Implementation: Export processed structures with properties.
- Assessment evidence: Students should verify metadata survives.


### Exercise 6: Molecule Visualization
- Objective: Use viewers for structural quality control.
- Orange workflow: `File -> Mol Viewer / Mol 3D Viewer`
- Implementation: Inspect problematic entries visually.
- Assessment evidence: Students should identify visual QC limitations.


### Exercise 7: R-Group Decomposition
- Objective: Analyze substituent patterns in a series.
- Orange workflow: `File -> R-Group Decomposition -> Data Table`
- Implementation: Map substituents to a common core.
- Assessment evidence: Students should interpret R-groups chemically.


### Exercise 8: Matched Molecular Pairs
- Objective: Find transformations associated with property changes.
- Orange workflow: `File -> Matched Molecular Pairs`
- Implementation: Review transformations and activity deltas.
- Assessment evidence: Students should connect MMPs to design hypotheses.


### Exercise 9: Reaction Enumeration
- Objective: Generate virtual products from rules and reactants.
- Orange workflow: `Reactants + Rules -> Reaction Enumerator`
- Implementation: Inspect enumerated products and failures.
- Assessment evidence: Students should document reaction scope assumptions.


### Exercise 10: Pharmacophore Fingerprint Search
- Objective: Search by pharmacophore-like feature patterns.
- Orange workflow: `File -> PharmaFP Search -> Data Table`
- Implementation: Compare structural vs pharmacophoric similarity.
- Assessment evidence: Students should explain feature abstraction.


### Exercise 11: Integrated QC Pipeline
- Objective: Design a reliable preparation pipeline for a library.
- Orange workflow: `Standardizer -> Filters -> Fingerprints -> Viewer -> Writer`
- Implementation: Create a QC table and export accepted compounds.
- Assessment evidence: Students should document every exclusion.


### Exercise 12: Capstone Library Design
- Objective: Design a focused screening library.
- Orange workflow: `Multiple workflow branches`
- Implementation: Deliver selected set and design rationale.
- Assessment evidence: Students should balance diversity, motifs, and constraints.
