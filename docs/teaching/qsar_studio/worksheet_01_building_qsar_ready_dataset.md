# Worksheet 01 — Building a QSAR-Ready Dataset

## Context

Raw bioactivity tables usually contain measurements, not directly model-ready
compounds. The same compound can appear multiple times, different endpoint types
may be mixed, and activity values may use different units.

## Intended learning outcomes

By the end of this worksheet, students can:

- identify required QSAR curation columns;
- convert concentration values to pActivity;
- explain why inequality records are often excluded from beginner QSAR models;
- aggregate duplicate compound measurements.

## Data

Use:

```text
examples/qsar_studio/qsar_dataset_builder_demo.csv
```

## Orange workflow

```text
Molecule Import Hub → QSAR Dataset Builder → Data Table
```

Recommended settings:

- SMILES: `canonical_smiles`
- Name/ID: `compound_id`
- Activity: `standard_value`
- Unit: `standard_units`
- Relation: `standard_relation`
- Endpoint: `standard_type`
- Keep endpoint: `IC50`
- Relations: `Exact values only`
- Duplicate aggregation: `median`

## Tasks

1. Build the QSAR-ready table.
2. Open the `Dataset Summary` output.
3. Open the `Rejected Records` output.
4. Explain why the invalid SMILES and inequality record were rejected.
5. Find the duplicate pyridine measurements and inspect the aggregated value.

## Evidence of learning

Students submit a short curation note containing:

- input record count;
- rejected record count;
- prepared compound count;
- endpoint used;
- aggregation method;
- two examples of rejected records and reasons.
