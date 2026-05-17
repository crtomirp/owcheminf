# Worksheet 05: Interpretable MLR Model Selection

**Estimated time:** 90–120 min  
**Level:** intermediate to advanced  
**Main widgets:** Mol Descriptors 2, MLR Model Selection, Data Table

## Context

Multiple linear regression can be easier to interpret than complex machine-learning models, especially when the descriptor set is carefully filtered.

## Intended learning outcomes

Students will be able to:

1. build an interpretable MLR model,
2. inspect selected descriptors and coefficients,
3. discuss multicollinearity and VIF,
4. relate descriptor signs to chemical hypotheses.

## Orange workflow

```text
File → Mol Descriptors 2 → MLR Model Selection
MLR Model Selection → Coefficients → Data Table
MLR Model Selection → Train Results → Data Table
```

## Student tasks

1. Generate molecular descriptors.
2. Build an MLR model.
3. Inspect selected descriptors.
4. Inspect coefficient signs.
5. Identify descriptors that may be chemically interpretable.
6. Discuss whether the model is overfit.

## Guiding questions

- Why should descriptor count be limited in small datasets?
- What does a positive coefficient suggest?
- What does a negative coefficient suggest?
- Why can correlated descriptors make interpretation unreliable?
- What is the role of VIF?

## Expected output

A coefficients table and train/test prediction tables.
