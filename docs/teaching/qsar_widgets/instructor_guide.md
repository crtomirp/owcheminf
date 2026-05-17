# Instructor Guide: QSAR Widgets

## Recommended setup

```bash
conda activate owcheminf
cd cinf
pip install -e .
orange-canvas
```

Recommended smoke tests:

```bash
pytest tests/test_fingerprints_phase1.py tests/test_mol_standardizer_phase1.py -q
pytest tests/test_cyclic_registry_fingerprint_phase2.py tests/test_cyclic_registry_validator_phase21.py -q
```

## Teaching strategy

Use small datasets first. The objective is workflow literacy: representation, descriptor calculation, splitting, modelling, validation, interpretation, and reporting.

## Common misconceptions

- High training R² is not proof of predictive performance.
- Random splits can overestimate performance when close analogues are in both train and test sets.
- More descriptors are not always better.
- Descriptor selection should not use the test set.
- A predictive descriptor is not automatically a causal mechanism.

## Suggested grading

| Criterion | Points |
|---|---:|
| Correct data preparation | 20 |
| Correct workflow construction | 20 |
| Validation reasoning | 20 |
| Chemical interpretation | 25 |
| Report clarity | 15 |
