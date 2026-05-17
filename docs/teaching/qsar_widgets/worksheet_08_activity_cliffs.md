# Worksheet 08: Activity Cliff Detection

**Estimated time:** 90 min  
**Level:** intermediate to advanced  
**Main widgets:** Fingerprint Generator, Activity Cliff Finder, Data Table

## Context

An activity cliff occurs when two structurally similar molecules have very different activities or properties. Activity cliffs are important because they reveal local structure-activity relationships.

## Intended learning outcomes

Students will be able to:

1. define an activity cliff,
2. find similar molecule pairs with large property differences,
3. interpret possible structural causes,
4. discuss why activity cliffs are challenging for QSAR models.

## Orange workflow

```text
File → Fingerprint Generator → Activity Cliff Finder → Data Table
```

Also inspect:

```text
Activity Cliff Finder → Cliff Compounds → Data Table
Activity Cliff Finder → Scaffold Summary → Data Table
```

## Student tasks

1. Generate fingerprints.
2. Configure the activity/property column.
3. Run `Activity Cliff Finder`.
4. Identify the top cliff pair.
5. Compare the two SMILES strings.
6. Propose a chemical explanation or a data-quality concern.

## Guiding questions

- What similarity threshold defines “similar enough”?
- What activity difference defines a cliff?
- Could an apparent activity cliff be caused by noisy data?
- Why do cliffs reduce model smoothness?

## Expected output

A table of cliff pairs and a table of compounds involved in cliffs.
