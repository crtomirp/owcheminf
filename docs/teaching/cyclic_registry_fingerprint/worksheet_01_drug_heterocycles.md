# Worksheet 01 — Identifying heterocycles in drug-like molecules

## Context

Students use the widget to detect heterocycles and functionalized ring systems in a small drug-like set.

**Keywords:** drug discovery, heterocycles, registry matches

## Intended learning outcomes

After completing this worksheet, students should be able to:

1. Build an Orange workflow using the **Cyclic Registry Fingerprint** widget.
2. Explain the difference between the 4096-bit fingerprint table and the **Matched Registry Entries** table.
3. Interpret at least three registry matches in chemical language.
4. Identify one limitation of fingerprint-based interpretation.

## Recommended input data

Use the packaged example file:

```text
examples/cyclic_registry_fingerprint/cyclic_registry_training_set.csv
```

Alternatively, prepare a CSV file with at least these columns:

```text
name,smiles,class
```

## Orange workflow

```text
File → Cyclic Registry Fingerprint → Matched Registry Entries → Data Table
```

Steps:

1. Open Orange Canvas.
2. Load the input file with the **File** widget.
3. Add **Cyclic Registry Fingerprint** from `Cheminf - Processing`.
4. Select the SMILES column.
5. Send `Fingerprints` to **Data Table** and `Matched Registry Entries` to **Data Table**.
6. Inspect both outputs and answer the questions below.

## CLI alternative without Orange

```bash
owcheminf-cyclic-registry-fingerprint \
  examples/cyclic_registry_fingerprint/cyclic_registry_training_set.csv \
  --smiles-column smiles \
  --name-column name \
  --out-prefix outputs/worksheet_01_drug_heterocycles_crfp \
  --write-json
```

Inspect:

```text
outputs/worksheet_01_drug_heterocycles_crfp.active_bits.csv
outputs/worksheet_01_drug_heterocycles_crfp.matches.csv
outputs/worksheet_01_drug_heterocycles_crfp.summary.json
```

## Student tasks

1. Count how many molecules produce at least one registry match.
2. Identify the three most frequent registry entry names in the matches table.
3. Select one molecule and explain its most important matched motifs.
4. Compare a matched registry bit with a Morgan bit. Which one is easier to explain?
5. Write a short paragraph describing one limitation of the analysis.

## Guiding questions

- Which molecules contain nitrogen heterocycles?
- Which molecules contain fused heterocycles?
- Which registry entries are chemically expected?
- Which matches are surprising and require checking?

## Expected outputs

The activity should produce:

- a 4096-bit fingerprint table,
- an interpretable matched-entry table,
- a short written interpretation of the chemical motifs,
- a brief statement on limitations such as bit collisions, tautomerism, protonation state, aromatic/kekulized representation, or registry coverage.

## Assessment rubric

| Criterion | Excellent | Satisfactory | Needs improvement |
|---|---|---|---|
| Workflow construction | Correct workflow and correct outputs | Minor workflow issues | Output missing or wrong widget used |
| Chemical interpretation | Motifs interpreted in chemical language | Basic but correct interpretation | Mostly mechanical listing of matches |
| Use of matched-entry table | Uses entry ID/name/section/match count | Uses only entry names | Does not use matched-entry output |
| Critical reflection | Clearly discusses limitations | Mentions one limitation | No limitation discussed |

## Teacher notes

Emphasize that the fingerprint has two complementary roles. The full bit vector is useful for machine learning, while the matched-entry table is useful for interpretation. Students should not assume that every active bit is uniquely interpretable, because registry entries can collide into the same bit and Morgan bits are hashed.
