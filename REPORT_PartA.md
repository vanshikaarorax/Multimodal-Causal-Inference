# MemoLogs — AI/ML Take-Home

This repository contains the complete implementation for **Part A (Persona Inference)** and **Part B (Revenue Lift Estimation)**, including source code, evaluation scripts, tests, notebooks, reports, and supporting configuration.

## 1. Full Project Architecture

```text
info_PartB.txt
├── Makefile
├── notebooks/
│   └── partB.ipynb
├── personas.yaml
├── pyproject.toml
├── README_PartA.md
├── README_PartB.md
├── README.md
├── REPORT.md
├── requirements.txt
├── scripts/
│   ├── analyze_all_pairwise.py
│   ├── build_duplicate_clusters.py
│   ├── create_duplicate_dev_set.py
│   ├── evaluate_abstention.py
│   ├── evaluate_duplicates.py
│   ├── inspect_duplicate_clusters.py
│   ├── run_partB.py
│   ├── save_persona_model.py
│   └── train_persona_models.py
├── srcA/
│   ├── __init__.py
│   ├── abstention.py
│   ├── classifier.py
│   ├── cli.py
│   ├── clustering.py
│   ├── config.py
│   ├── data.py
│   ├── duplicate.py
│   ├── embeddings.py
│   ├── embeddings2.py
│   ├── evaluate.py
│   ├── evidence.py
│   ├── knn_model.py
│   ├── notebooks/embeddings.ipynb
│   ├── pipeline.py
│   ├── rationale.py
│   ├── schema.py
│   └── split.py
├── srcB/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── design.py
│   ├── diagnostics.py
│   ├── estimators.py
│   ├── evaluate.py
│   ├── notebooks/partB.ipynb
│   ├── pipeline.py
│   ├── placebo.py
│   └── trust.py
├── TASK.md
└── tests/
    ├── test_abstention.py
    ├── test_evidence.py
    ├── test_partB_design.py
    ├── test_schema.py
    ├── test_split.py
    └── test_trust.py
```

`__pycache__`, generated artifacts, embeddings, images, and other large runtime files are omitted from the architecture above for readability.

---

# 2. Part A — Persona Inference

Part A builds an end-to-end system that uses multimodal creative embeddings, duplicate detection, leakage-safe persona prediction, abstention, evidence retrieval, and grounded LLM rationales.

### Code Architecture

| Module | Purpose |
|---|---|
| `srcA/data.py` | Loads creatives, labels, personas, and supporting data. |
| `srcA/config.py` | Central configuration for models, thresholds, paths, and embedding settings. |
| `srcA/embeddings.py` | Generates and caches image/text CLIP embeddings and weighted multimodal embeddings. |
| `srcA/embeddings2.py` | Additional embedding/model experimentation. |
| `srcA/duplicate.py` | Computes embedding similarity and identifies near-duplicate creatives. |
| `srcA/clustering.py` | Builds duplicate clusters used for leakage-safe evaluation. |
| `srcA/split.py` | Splits data by duplicate clusters so near-duplicates cannot cross train/dev/test. |
| `srcA/knn_model.py` | kNN baseline for persona prediction. |
| `srcA/classifier.py` | One-vs-rest logistic-regression multi-label persona classifier. |
| `srcA/evaluate.py` | Computes overall, per-persona, and selective prediction metrics. |
| `srcA/abstention.py` | Applies suggest / suggest-with-caution / abstain decisions. |
| `srcA/evidence.py` | Retrieves similar confirmed creatives and persona evidence. |
| `srcA/rationale.py` | Generates evidence-grounded LLM rationales with deterministic fallback. |
| `srcA/schema.py` | Defines and validates the structured output schema using Pydantic. |
| `srcA/pipeline.py` | Connects prediction, abstention, evidence, rationale, and validation. |
| `srcA/cli.py` | Exposes the final persona suggestion pipeline through the CLI. |

### Part A Supporting Files

- **`README_PartA.md`** — Part A usage and implementation notes.
- **`REPORT.md`** — concise technical report covering methodology, experiments, evaluation, limitations, and final decisions.
- **`TASK.md`** — original assignment/task specification.
- **`scripts/`** — reproducible utilities for embedding analysis, duplicate evaluation/clustering, model training, and abstention evaluation.
- **`srcA/notebooks/embeddings.ipynb`** — exploratory embedding analysis.
- **`tests/`** — unit tests covering abstention, evidence, schema validation, and leakage-safe splitting.
- **`personas.yaml`** — persona definitions used by the system.
- **`Makefile`** — common commands for tests, evaluation, and CLI smoke testing.
- **`pyproject.toml` / `requirements.txt`** — project metadata and dependency specifications.

### Part A End-to-End Flow

```text
Creative
   ↓
Image + Text
   ↓
CLIP embeddings
   ↓
Weighted multimodal fusion
   ↓
Near-duplicate detection / clustering
   ↓
Cluster-safe train/dev/test split
   ↓
Persona prediction
   ↓
Abstention decision
   ↓
Evidence retrieval
   ↓
Grounded LLM rationale / template fallback
   ↓
Pydantic schema validation
   ↓
CLI JSON output
```

---

# 3. Part B — Revenue Lift Estimation

Part B extends the project from persona inference to estimating the causal revenue impact of targeting decisions using geo-level experimental/observational data and diagnostic checks.

### Code Architecture

| Module | Purpose |
|---|---|
| `srcB/data.py` | Loads and prepares Part B datasets. |
| `srcB/config.py` | Central Part B configuration. |
| `srcB/design.py` | Defines the experimental/causal design and analysis setup. |
| `srcB/estimators.py` | Implements revenue-lift estimators. |
| `srcB/diagnostics.py` | Runs diagnostics and checks assumptions/results. |
| `srcB/placebo.py` | Runs placebo tests to check for spurious effects. |
| `srcB/trust.py` | Implements trust/validity checks for the estimated lift. |
| `srcB/evaluate.py` | Evaluates and summarizes Part B results. |
| `srcB/pipeline.py` | Orchestrates the complete Part B workflow. |
| `srcB/cli.py` | Provides the Part B command-line interface. |

### Part B Supporting Files

- **`README_PartB.md`** — Part B methodology and execution notes.
- **`srcB/notebooks/partB.ipynb`** — analysis notebook for exploration and diagnostics.
- **`scripts/run_partB.py`** — executable Part B runner.
- **`tests/test_partB_design.py`** — tests for the causal/experimental design.
- **`tests/test_trust.py`** — tests for trust and validity checks.
- **`geo_panel.csv`** — geo-level panel data used by the Part B analysis.
- **`README.md`** — this repository-level overview.

### Part B End-to-End Flow

```text
Geo-level panel data
        ↓
Data preparation
        ↓
Experimental / causal design
        ↓
Revenue-lift estimation
        ↓
Diagnostics + placebo checks
        ↓
Trust / validity checks
        ↓
Evaluation
        ↓
Final lift estimates
```

---

# 4. Reproducibility & Validation

The repository is organized so that the implementation, experiments, evaluation, and tests are separated but connected through the two pipelines.

Typical validation:

```bash
make all
```

Part A CLI smoke test:

```bash
MOCK_LLM=1 python -m srcA.cli --creative-id c_0412 --mock-llm
```

The `MOCK_LLM=1` path allows deterministic execution without requiring an API key.

## Design Principles

- **Multimodal:** image and text signals are combined rather than relying on a single modality.
- **Leakage-safe:** duplicate clusters are kept within a single train/dev/test split.
- **Selective:** the system can abstain when model evidence is weak.
- **Grounded:** rationales are restricted to retrieved evidence and validated against a schema.
- **Reproducible:** configuration, scripts, tests, and cached artifacts are separated from exploratory notebooks.
- **Tested:** both Part A and Part B contain focused automated tests for important correctness and trust properties.

For detailed methodology and experimental results, see `REPORT_PART-A.md`,`REPORT_PART-B.md`, `README_PartA.md`, and `README_PartB.md`.
