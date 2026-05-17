# Web Phase A Backend Backlog

This document turns the higher-level web migration plan into a concrete Phase A implementation backlog.

Primary objective:

- stand up a reusable Python API layer on top of existing `chemcore/services`
- define the first stable workflow and dataset contracts
- keep Orange widgets working unchanged while web foundations are built

Related architecture note:

- [web_workflow_migration_plan.md](/Users/crtomir/Desktop/cinf/docs/development/web_workflow_migration_plan.md)

## Phase A scope

Phase A should not build the full web UI yet.

It should deliver:

- a backend service package
- stable JSON request/response schemas
- first workflow serialization format
- first 4 to 6 API-backed compute nodes
- repeatable tests for API and service contracts

Out of scope for Phase A:

- full React workflow canvas
- HPC submission
- browser-side RDKit WASM execution
- user accounts / multi-user tenancy

## Recommended Phase A stack

- `FastAPI` for HTTP API
- `Pydantic` for contracts
- `Uvicorn` for local serving
- existing `chemcore/services` as execution core
- `pytest` or `unittest` contract tests
- `Parquet` or `Arrow` for larger tabular payloads later

## Proposed repository layout

This can live inside the current repo first.

```text
web/
  backend/
    app/
      __init__.py
      main.py
      api/
        health.py
        standardization.py
        fingerprints.py
        similarity.py
        qsar.py
        workflows.py
      schemas/
        common.py
        datasets.py
        nodes.py
        workflows.py
      execution/
        registry.py
        dispatch.py
        standardization_runner.py
        fingerprint_runner.py
        similarity_runner.py
        qsar_runner.py
      adapters/
        chemcore_standardization.py
        chemcore_fingerprints.py
        chemcore_similarity.py
        chemcore_qsar.py
      storage/
        dataset_store.py
        workflow_store.py
      tests/
```

## Initial backend modules to build

### 1. App bootstrap

Files:

- `web/backend/app/main.py`
- `web/backend/app/api/health.py`

Purpose:

- start service
- expose `/health`
- expose `/version`

## First API surface

The first API should focus on a minimal but representative workflow.

### Endpoint 1: standardize molecules

Route:

- `POST /api/v1/standardize`

Backed by:

- [`mol_standardizer.py`](/Users/crtomir/Desktop/cinf/src/chem_inf_widgets/chemcore/services/mol_standardizer.py)
- [`rdkit_safe.py`](/Users/crtomir/Desktop/cinf/src/chem_inf_widgets/chemcore/services/rdkit_safe.py)

Request shape:

```json
{
  "dataset": {
    "columns": ["mol_id", "input_smiles"],
    "rows": [
      {"mol_id": "m1", "input_smiles": "c1ccncc1"}
    ]
  },
  "settings": {
    "neutralize": true,
    "largest_fragment": true,
    "canonicalize": true
  }
}
```

Response shape:

```json
{
  "dataset": {
    "columns": ["mol_id", "input_smiles", "standardized_smiles", "inchikey", "qc_status"],
    "rows": []
  },
  "summary": {
    "n_rows": 0,
    "n_valid": 0,
    "n_invalid": 0
  },
  "issues": []
}
```

### Endpoint 2: generate fingerprints

Route:

- `POST /api/v1/fingerprints`

Backed by:

- [`fingerprints.py`](/Users/crtomir/Desktop/cinf/src/chem_inf_widgets/chemcore/descriptors/fingerprints.py)

Request shape:

```json
{
  "dataset": {
    "columns": ["mol_id", "canonical_smiles"],
    "rows": []
  },
  "settings": {
    "fingerprint_type": "morgan",
    "radius": 2,
    "n_bits": 2048,
    "use_chirality": false
  }
}
```

### Endpoint 3: similarity search

Route:

- `POST /api/v1/similarity-search`

Backed by:

- [`substructure_search_service.py`](/Users/crtomir/Desktop/cinf/src/chem_inf_widgets/chemcore/services/substructure_search_service.py)
- [`ow_similarity_search.py`](/Users/crtomir/Desktop/cinf/src/chem_inf_widgets/widgets/ow_similarity_search.py) behavior as reference only

Request shape:

```json
{
  "query_dataset": {
    "columns": ["mol_id", "canonical_smiles"],
    "rows": []
  },
  "reference_dataset": {
    "columns": ["mol_id", "canonical_smiles"],
    "rows": []
  },
  "settings": {
    "metric": "tanimoto",
    "cutoff": 0.6,
    "top_k": 10
  }
}
```

### Endpoint 4: QSAR regression

Route:

- `POST /api/v1/qsar/regression`

Backed by:

- [`qsar_regression_service.py`](/Users/crtomir/Desktop/cinf/src/chem_inf_widgets/chemcore/services/qsar_regression_service.py)

Request shape:

```json
{
  "dataset": {
    "columns": [],
    "rows": []
  },
  "external_dataset": null,
  "settings": {
    "selected_algorithm": 0,
    "normalization_method": 0,
    "imputation_method": 1,
    "cv_folds": 5,
    "test_size": 0.3,
    "tuning_method": 0,
    "n_iter": 10,
    "enable_feature_selection": false,
    "num_features": 10,
    "enable_applicability_domain": true
  }
}
```

### Endpoint 5: workflow execute

Route:

- `POST /api/v1/workflows/execute`

Purpose:

- execute a small JSON workflow definition without a UI
- validate node contracts early
- prove orchestration before building the canvas

## First workflow schema

Suggested schema file:

- `web/backend/app/schemas/workflows.py`

Suggested minimal structure:

```json
{
  "version": "1.0",
  "workflow_id": "demo-1",
  "nodes": [
    {
      "id": "input-1",
      "type": "dataset_input",
      "category": "Cheminf - Data",
      "settings": {},
      "execution_target": "backend"
    }
  ],
  "edges": [
    {
      "source": "input-1",
      "source_port": "dataset",
      "target": "std-1",
      "target_port": "dataset"
    }
  ],
  "datasets": {},
  "metadata": {
    "title": "Demo workflow"
  }
}
```

## First dataset schema

Suggested schema file:

- `web/backend/app/schemas/datasets.py`

Suggested model:

```json
{
  "columns": [
    {
      "name": "canonical_smiles",
      "dtype": "string",
      "role": "smiles"
    }
  ],
  "rows": [
    {
      "canonical_smiles": "c1ccncc1"
    }
  ],
  "metadata": {
    "row_count": 1
  }
}
```

Recommended required column roles:

- `smiles`
- `name`
- `target`
- `descriptor`
- `meta`

## First 5 web nodes

These should be implemented first because they exercise the main system without being too broad.

### Node 1: Dataset Input

Responsibilities:

- ingest CSV/JSON payload
- validate required columns
- assign dataset metadata

Execution target:

- backend

### Node 2: Standardizer

Responsibilities:

- parse and standardize molecules
- emit stable chemistry identifiers
- return conversion summary

Execution target:

- backend first

### Node 3: Fingerprint Generator

Responsibilities:

- generate fingerprint matrix
- emit bit metadata
- support Morgan first

Execution target:

- backend first

### Node 4: Similarity Search

Responsibilities:

- compare query dataset to reference dataset
- emit pairwise results
- support threshold and top-k

Execution target:

- backend first

### Node 5: QSAR Regression

Responsibilities:

- prepare descriptor matrix
- train and validate model
- emit metrics and prediction tables

Execution target:

- backend only

## Execution registry

Suggested file:

- `web/backend/app/execution/registry.py`

This registry should map:

- node type
- schema
- runner
- default execution target

Example registry idea:

```python
NODE_REGISTRY = {
    "standardizer": StandardizerRunner,
    "fingerprint_generator": FingerprintRunner,
    "similarity_search": SimilarityRunner,
    "qsar_regression": QSARRegressionRunner,
}
```

## Adapter strategy

Do not call Orange widgets from the backend.

Instead, each runner should call an adapter that wraps `chemcore/services`.

Example:

- `chemcore/services/qsar_regression_service.py`
- `web/backend/app/adapters/chemcore_qsar.py`

Why this layer helps:

- isolates web-facing request shapes
- avoids coupling HTTP contracts to Orange assumptions
- makes later HPC routing easier

## Backlog by priority

### Priority 1: contracts and app shell

- create `web/backend/app/main.py`
- add `/health`
- add Pydantic dataset schema
- add Pydantic workflow schema
- add node registry skeleton

### Priority 2: first service endpoints

- implement `/standardize`
- implement `/fingerprints`
- implement `/similarity-search`
- implement `/qsar/regression`

### Priority 3: workflow execution

- implement `/workflows/execute`
- validate DAG structure
- execute nodes in dependency order
- return per-node results and issues

### Priority 4: tests

- schema validation tests
- endpoint contract tests
- service adapter tests
- one end-to-end workflow execution test

## Suggested test cases

### Standardization

- valid SMILES
- invalid SMILES
- mixed valid and invalid rows
- neutralization on/off

### Fingerprints

- small dataset
- duplicate molecules
- empty dataset

### Similarity

- one query, many references
- threshold filtering
- top-k truncation

### QSAR

- valid descriptor dataset
- missing target error
- too few rows error
- external dataset optional

### Workflow execution

- linear 3-node workflow
- branching workflow
- invalid edge reference
- unsupported node type

## Recommended first milestone

Deliver a CLI-free backend demo that can run this workflow through one API call:

```text
Dataset Input
-> Standardizer
-> Fingerprint Generator
-> Similarity Search
```

Then deliver a second backend demo:

```text
Dataset Input
-> QSAR Regression
```

That is enough to validate:

- chemistry service reuse
- dataset contracts
- node orchestration
- API design

## Recommended second milestone

Once the above works:

- add a minimal frontend that can submit workflow JSON
- render node statuses
- show returned datasets and summaries

Only after that should the full interactive canvas begin.
