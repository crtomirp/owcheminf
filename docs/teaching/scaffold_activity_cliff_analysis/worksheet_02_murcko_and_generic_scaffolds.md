# Murcko and Generic Scaffold Analysis

## Context

Murcko scaffolds retain ring systems and linkers. Generic Murcko scaffolds remove atom and bond specificity, making broader scaffold families.

## Intended learning outcomes

Students can:

- distinguish exact Murcko and generic Murcko scaffolds,
- explain why generic scaffolds create broader groups,
- interpret scaffold summary tables.

## Orange workflow

```text
File → Scaffold Analysis → Scaffold Summary → Data Table
```

## Tasks

1. Run **Scaffold Analysis** with `Exact Murcko`.
2. Save or inspect the `Scaffold Summary` output.
3. Repeat with `Generic Murcko`.
4. Compare the number of scaffold groups.

## Guiding questions

1. Which mode gives fewer scaffold groups?
2. Which mode is more chemically specific?
3. Which mode is more useful for broad train/test splitting?
4. What information is lost when using generic scaffolds?

## Expected output

Students should understand that exact scaffolds are more specific, while generic scaffolds may be better for conservative validation.

## Assessment rubric

| Criterion | Basic | Good | Excellent |
|---|---|---|---|
| Workflow execution | Runs the main widgets | Correct settings and outputs | Clear, reproducible workflow |
| Chemical interpretation | Minimal description | Correct scaffold/cliff explanation | Insightful SAR reasoning |
| Validation discussion | Mentions train/test split | Explains random vs scaffold split | Connects validation, AD, and cliffs |
| Reporting | Fragmentary notes | Clear short answers | Publication-style transparency |
