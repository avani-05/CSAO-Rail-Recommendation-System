# Source Code Structure

This folder contains the modular implementation of the two-stage
cart-based recommendation system.

While experimentation and analysis are performed in notebooks,
this directory contains reusable, production-oriented components
that mirror the full pipeline.

---

## Architecture Overview

The system follows a two-stage design:

1. Candidate Retrieval (high recall)
2. Learned Ranking (precision optimization)

The `src/` layer separates data loading, feature construction,
retrieval logic, ranking, evaluation, and inference.

This separation improves:

- Reproducibility
- Maintainability
- Deployability
- System clarity

---

## File Descriptions

### `data_loader.py`

Handles structured loading of datasets from:

- `data/raw/`
- `data/enriched/`
- `data/model_input/`

Encapsulates path management and keeps notebooks clean.

---

### `feature_pipeline.py`

Implements feature engineering logic:

- Cart state features
- Aggregated user statistics
- Item-level attributes
- Cross features
- Label construction (next-step-only)

This module transforms candidate-level data into model-ready matrices.

---

### `retrieval_engine.py`

Implements Stage 1 candidate generation:

- Item-to-item co-occurrence
- Context-aware popularity (meal-time buckets)
- Hard constraints (same restaurant)
- Cart exclusion logic

Output: ~25–30 candidates per cart state.

Designed for high recall and low latency.

---

### `ranking_model.py`

Encapsulates the LightGBM ranking model:

- Training
- Prediction
- Save/load model artifact
- Early stopping support

Binary objective is used to model:

P(next_item | cart_state)

Predicted probabilities are used as ranking scores.

---

### `evaluation.py`

Implements grouped ranking metrics:

- NDCG@K
- Precision@K
- Recall@K

Groups are defined by:

(session_id, step_number)

---

### `ab_testing.py`

Contains logic for offline A/B simulation:

- Score uplift
- Attach-rate proxy lift
- Visibility improvement analysis

Used to translate model performance into business impact.

---

### `inference_pipeline.py`

Provides a lightweight inference wrapper:

- Loads trained model
- Scores candidate feature matrix
- Returns ranked items

This structure makes it straightforward to expose the model via an API.

---

## Why This Folder Exists

Notebooks are excellent for experimentation.

However, production systems require:

- Modular design
- Clear responsibility separation
- Testable components
- Deployment compatibility

This folder bridges experimentation and production readiness.

---

## Relationship to Notebooks

Notebook → Corresponding Module

- 01_split_and_feature_engineering.ipynb → feature_pipeline.py
- 02_candidate_generation.ipynb → retrieval_engine.py
- 03_model_training_lightgbm.ipynb → ranking_model.py
- 04_ab_testing_simulation.ipynb → evaluation.py & ab_testing.py

This ensures the research workflow maps cleanly to reusable code.
