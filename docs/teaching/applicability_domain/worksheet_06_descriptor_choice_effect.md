# Worksheet 06 — Descriptor Choice Changes the Domain

## Context

Applicability domain is defined in descriptor space. If the descriptors change, the domain changes.

## Intended learning outcomes

Students will be able to:

1. Explain why descriptor choice affects AD results.
2. Compare AD using simple physicochemical descriptors and larger descriptor sets.
3. Recognize that AD is not an intrinsic property of a molecule alone.

## Orange workflow

Workflow A:

```text
File → Applicability Domain
```

using existing descriptor columns:

```text
MW, LogP, TPSA, HBD, HBA, RotB, AromaticRings
```

Workflow B:

```text
File → Mol Descriptors 2 → Applicability Domain
```

## Student tasks

1. Run AD using only the provided descriptor columns.
2. Run AD after calculating additional descriptors.
3. Compare `AD_in_domain` outcomes.
4. Identify compounds whose status changes.

## Guiding questions

1. Which descriptor set is more chemically informative?
2. Which descriptor set is more likely to suffer from high dimensionality?
3. How many reference compounds are needed for a reliable high-dimensional domain?
4. Should AD be reported together with the descriptors used?

## Expected conclusion

The applicability domain must always be described together with the descriptor space and preprocessing used to define it.
