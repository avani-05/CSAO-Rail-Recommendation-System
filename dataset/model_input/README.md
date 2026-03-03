
# Model Input (Stage 2 Ranking)

This folder contains the final model-ready datasets used to train and
evaluate the LightGBM ranking model.

This stage consumes the candidate pool generated in Stage 1 and
constructs fully joined, labeled feature matrices for supervised learning.

---

## Purpose of This Stage

Given a cart state at step N and a candidate pool of items:

- Build all contextual, user, item, and restaurant features
- Construct the next-step label
- Create grouped ranking queries
- Produce final matrices ready for LightGBM training

This is the input to the Stage 2 ranking model.

---

## Files in This Folder

### 1️⃣ rank_train_sample.csv  
### 2️⃣ rank_test_sample.csv  

These contain the expanded candidate-level dataset before final
feature filtering.

Each row represents:

(session_id, step_number, candidate_item_id)

Columns include:
- Label: `label_addon_added`
- Retrieval signals: `retrieval_score`, `src_cooc`, `src_ctx`, `src_rule`
- Cart state features
- User features
- Item features
- Restaurant features

These files preserve all joined features prior to feature selection.

---

### 3️⃣ lgbm_train_features_sample.csv  
### 4️⃣ lgbm_test_features_sample.csv  

These are the final filtered feature matrices used for model training.

Processing applied:

- Removed identifiers (session_id, candidate_item_id, etc.)
- Dropped low-signal and object columns
- Removed redundant one-hot encodings
- Applied numeric casting
- Ensured no duplicate features

Final feature count: ~47 features

These files are directly consumed by LightGBM.

---

## Label Definition

Label: `label_addon_added`

Definition:
For cart state at step N, label = 1 only for the item added at step N+1.

Each (session_id, step_number) group has:
- Exactly 1 positive
- ~25 negative candidates

This creates a clean ranking objective:
Predict which candidate is most likely to be added next.

---

## Group Structure

Each query group is defined by:

(session_id, step_number)

Average candidates per group: ~25–30  
Exactly one positive per group.

This structure supports:

- NDCG@K evaluation
- Precision@K
- Recall@K
- Ranking-based metrics

---

## Feature Categories

Features include:

### Retrieval Signals
- retrieval_score
- src_cooc
- src_ctx
- src_rule

### Cart Context
- cart_item_count
- cart_total_value
- missing_beverage_flag
- missing_dessert_flag
- step_decay

### User Features
- order_count_90d
- recency_days
- avg_order_value
- veg_preference_ratio
- price_sensitivity_score

### Item Features
- price
- popularity_score
- historical_attach_rate
- effective_category_*

### Cross Features
- retrieval_x_attach
- retrieval_x_popularity
- missing_bev_x_retrieval
- x_price_sens_price

These engineered features allow the model to learn interaction effects
between context, user behavior, and item characteristics.

---

## Model Choice Rationale

We use a binary LightGBM classifier rather than LambdaRank.

Why?

- Each group contains ~1 positive and ~25 negatives.
- Pairwise ranking objectives produce unstable gradients
  when positives are sparse.
- Binary cross-entropy directly models P(next_item | cart_state).

The predicted probability is then used as a ranking score at inference.

---

## Production Considerations

- All joins are precomputed for efficiency.
- Final matrices are numeric and optimized for LightGBM.
- Grouped structure preserved for ranking evaluation.
- Compatible with real-time scoring.

This stage transforms retrieval candidates into a fully learned ranking system.
