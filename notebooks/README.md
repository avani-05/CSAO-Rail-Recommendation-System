
# Notebooks – End-to-End Pipeline

This folder contains the full experimental and modeling workflow
for the two-stage cart-based recommendation system.

Notebooks are organized sequentially and should be executed in order.

---

## Execution Order

1. 01_split_and_feature_engineering.ipynb  
2. 02_candidate_generation.ipynb  
3. 03_model_training_lightgbm.ipynb  
4. 04_ab_testing_simulation.ipynb  

Optional:
- DS_Enrichment.ipynb (LLM-based dataset enrichment)

Each notebook produces artifacts consumed by the next stage.

---

## Notebook Descriptions

### 01_split_and_feature_engineering.ipynb

Purpose:
- Train/test session split
- Data cleaning
- Cart state construction
- Feature engineering
- Next-step label definition

Outputs:
- Cleaned datasets in `data/enriched/`
- Intermediate feature datasets
- Label-aligned cart states

Key design choice:
Only the immediate next item is treated as positive
to avoid label leakage and ensure exactly one positive per group.

---

### 02_candidate_generation.ipynb

Purpose:
- Build item co-occurrence tables
- Compute context-aware popularity (meal-time buckets)
- Generate candidate pools per cart state
- Enforce restaurant-level constraints
- Ensure true next item is always included

Outputs:
- cooc_top.csv
- ctx_pop_top.csv
- Candidate-expanded dataset

Goal:
Maximize recall while keeping candidate pool bounded (~25–30).

---

### 03_model_training_lightgbm.ipynb

Purpose:
- Join retrieval features with contextual signals
- Perform feature selection
- Train LightGBM binary classifier
- Evaluate using grouped ranking metrics

Outputs:
- lgbm_train_features.csv
- lgbm_test_features.csv
- Trained model artifact (saved to `models/`)

Evaluation metrics:
- AUC
- NDCG@8
- Recall@8

Binary objective is used to model:
P(next_item | cart_state)

---

### 04_ab_testing_simulation.ipynb

Purpose:
- Simulate offline A/B comparison
- Compare retrieval-only vs learned ranking
- Estimate attach-rate proxy lift
- Translate metric gains into business impact

Outputs:
- Offline uplift tables
- Visibility improvement estimates

---

### DS_Enrichment.ipynb (Optional)

Purpose:
- LLM-based menu normalization
- Noise removal from item metadata
- Tag extraction (e.g., cuisine, category)
- Structured enrichment using Gemini API

Outputs:
- items_llm_tags.csv
- Cleaned item metadata used in feature engineering

---

## Why Notebooks Are Still Included

While the `src/` folder contains modular production-style code,
these notebooks provide:

- Transparent experimentation
- Step-by-step reasoning
- Metric tracking
- Reproducibility

They document the research-to-production transition.
