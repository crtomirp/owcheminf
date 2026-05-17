# Connecting Scaffold, Activity Cliff, QSAR, and Applicability Domain

## Context

Scaffold analysis, activity cliffs, QSAR validation, and applicability domain are complementary. Together they provide a stronger reliability assessment.

## Intended learning outcomes

Students can:

- combine scaffold and cliff information,
- explain why cliffs may reduce QSAR accuracy,
- relate scaffold novelty to applicability domain warnings.

## Orange workflow

```text
File
 → Mol Standardizer
 → Scaffold Analysis
 → Scaffold Splitter
 → QSAR Regression
 → Applicability Domain
```

Parallel analysis:

```text
File → Activity Cliff Finder → Cliff Pairs
```

## Tasks

1. Identify major scaffolds.
2. Create scaffold train/test split.
3. Identify activity cliff compounds.
4. Discuss which test compounds may be difficult to predict.
5. Relate difficult compounds to scaffold novelty or cliff membership.

## Guiding questions

1. Are cliff compounds more likely to be prediction errors?
2. Are new scaffolds outside the applicability domain?
3. Can a compound be inside AD but still difficult because of an activity cliff?
4. How should this be reported?

## Expected output

Students should understand that model reliability is multi-factorial.

## Assessment rubric

| Criterion | Basic | Good | Excellent |
|---|---|---|---|
| Workflow execution | Runs the main widgets | Correct settings and outputs | Clear, reproducible workflow |
| Chemical interpretation | Minimal description | Correct scaffold/cliff explanation | Insightful SAR reasoning |
| Validation discussion | Mentions train/test split | Explains random vs scaffold split | Connects validation, AD, and cliffs |
| Reporting | Fragmentary notes | Clear short answers | Publication-style transparency |
