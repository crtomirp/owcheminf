# Example Data: Scaffold and Activity Cliff Analysis

This folder contains small teaching datasets for Package C.

## Files

- `scaffold_activity_training_set.csv` — mixed scaffold families with synthetic pIC50 values.
- `activity_cliff_demo_set.csv` — same compounds, intended for Activity Cliff Finder demonstrations.
- `scaffold_split_demo_set.csv` — simplified columns for Scaffold Splitter exercises.

## Columns

- `name`: compound identifier.
- `smiles`: molecular structure.
- `scaffold_family` or `series`: approximate teaching label, not used by the scaffold algorithm.
- `pIC50`: synthetic log-potency value for teaching.
- `comment`: short explanation for selected compounds.

The pIC50 values are synthetic and should not be interpreted as real biological measurements.
