<div align="center">

# 🛒 CartNext - CSAO Rail Recommendation System

**Cart Super Add-On (CSAO) intelligent recommendation engine for food delivery**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Binary%20Ranker-009966?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![Gemini](https://img.shields.io/badge/Gemini%201.5%20Flash-LLM%20Enrichment-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Colab](https://img.shields.io/badge/Google%20Colab-Notebooks-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*Submitted for Zomathon — Problem Statement 2*

[Overview](#-overview) · [Results](#-results) · [Architecture](#-system-architecture) · [Notebooks](#-notebooks) · [Features](#-feature-engineering) · [Quickstart](#-quickstart) · [A/B Testing](#-ab-testing)

---

</div>

## 📌 Overview

CartNext is a **two-stage, real-time recommendation system** that intelligently suggests add-on items to food delivery customers based on their current cart composition, session context, and historical behaviour.

When a user adds items to their cart (e.g., Biryani → the system recommends Salan → after Salan is added, suggests Gulab Jamun → then Drinks), CartNext continuously updates the top-8 rail with the most contextually relevant next add-ons.

### The Problem

| Challenge | Description |
|---|---|
| **Cart context understanding** | Recognising incomplete meal patterns (main without beverage, single-item orders) |
| **Contextual relevance** | Recommendations must adapt to meal time, restaurant type, user segment, and geography |
| **Sequential updates** | Rail must refresh dynamically as items are added or removed |
| **Latency** | Predictions must be served within 200–300 ms at millions of requests/day |
| **Cold start** | New users, new restaurants, and tail items must still receive relevant recommendations |

### Our Solution

A **binary classification LightGBM ranker** trained on next-step cart labels, sitting on top of a three-source retrieval stage (co-occurrence, context-popularity, rule-fill). Candidate items are scored and the top-8 are served to the CSAO rail.

> **Why binary classification over LambdaRank?**  
> With ~25 candidates per query group and a 3.79% positive rate (1 positive per group), pairwise ranking objectives produce near-zero gradients. Binary cross-entropy handles this naturally, and the predicted probability is a valid ranking score at inference time — the same approach used in production at DoorDash, Instacart, and Zomato.

---

## 🏆 Results

### Model Performance vs Baselines

| Method | NDCG@5 | NDCG@8 | NDCG@10 | Prec@8 | Recall@8 | MRR |
|---|---|---|---|---|---|---|
| **LightGBM Ranker (ours)** | **0.4161** | **0.4556** | **0.4755** | **0.0895** | **0.7158** | **0.3941** |
| Baseline A: Retrieval Score | 0.3564 | 0.3987 | 0.4183 | 0.0799 | 0.6389 | 0.3485 |
| Baseline B: Popularity Score | 0.1564 | 0.2080 | 0.2370 | 0.0520 | 0.4161 | — |
| Baseline C: Historical Attach Rate | 0.2806 | 0.3209 | 0.3365 | 0.0710 | 0.5683 | — |

> All metrics evaluated on **20,246 test query groups** (533,556 rows). Statistical significance: p < 0.001 (paired t-test + Wilcoxon) for all primary metrics.

### Key Metrics at a Glance

```
AUC (test)       →  0.79290      NDCG@8 lift    →  +14.3% vs best baseline
Recall@8         →  71.58%       Hit Rate@1     →  21.33%
MRR              →  0.3941       Inference P99  →  0.119 ms
Coverage         →  100%         Cold-start     →  96.4% via rule-fill
```

### Business Impact Projections

| Metric | Control (Retrieval) | Treatment (LightGBM) | Lift |
|---|---|---|---|
| Acceptance Rate (Recall@8) | 63.89% | 71.58% | **+12.0%** |
| AOV Lift per Session | ₹48.99 | ₹61.48 | **+₹12.49** |
| AOV as % of baseline (₹450) | — | — | **+2.78%** |
| Daily Revenue Uplift (5M sessions) | — | — | **₹6.25 Cr/day** |

### Segment Performance (A/B Offline Simulation)

| Segment | NDCG@8 Lift | Recall@8 Lift |
|---|---|---|
| Dinner | +15.9% | +13.4% |
| Premium users | +15.0% | +12.0% |
| Late-night | +13.9% | +12.1% |
| Budget users | +13.4% | +13.1% |
| Lunch | +13.4% | +11.1% |
| Breakfast | +8.6% | +6.8% |

---

## 🏗 System Architecture

The system operates in three distinct layers:

### Layer 1 — Data Preparation

```
Raw Sources                    Data Preparation Layer
─────────────                  ──────────────────────────────────────────
User Data          ──►         Synthetic Data Generator
Session Cart Events ──►        ├── Data Cleaning
Restaurant Data    ──►         ├── Gemini API Enrichment
Menu Item Data     ──►         │   ├── Category Normalisation
Context Signals    ──►         │   ├── Meal Time Tags (multi-label)
                               │   └── Dish Subtype Inference
                               └── Feature Engineering
                                   ├── Item Attributes
                                   ├── User Historical Features
                                   ├── Cart State Features
                                   ├── Restaurant Features
                                   └── Context Features
                                        ↓
                               Feature Store
```

### Layer 2 — Two-Stage Ranking

```
Stage 1: Candidate Generation (retrieval)      Stage 2: Ranking Model
──────────────────────────────────────         ───────────────────────
Co-occurrence Retrieval      ──┐               Candidate Feature Join
Next-Step Label Logic        ──┼──► 25–30 ──►  Feature Matrix Builder
Rule-Based Fill              ──┤  Candidates   LightGBM Classifier
Context Popularity           ──┘               Scoring Engine
                                                    ↓
                                             Top 8–10 Recommendations
```

### Layer 3 — Online Inference & Monitoring

```
User Cart Event  ──►  Online Feature Fetch  ──►  Candidate Retrieval
──►  LightGBM Scoring  ──►  Ranked Output  ──►  CSAO Rail UI

Offline Metrics:  NDCG@K · Recall@K · AUC · Feature Importance
Online Monitoring: Attach Rate · AOV Lift · Cart Abandonment · Latency Guardrails
```

---

## 📁 Repository Structure

```
CartNext-Recommendation-System/
├── README.md                          ← you are here
│
├── notebooks/                         ← end-to-end Colab pipeline
│   ├── DS_Enrichment.ipynb            ← Gemini 1.5 Flash LLM enrichment
│   ├── 01_split_and_feature_engineering.ipynb
│   ├── 02_candidate_generation.ipynb
│   ├── 03_model_training_lightgbm.ipynb
│   └── 04_ab_testing_simulation.ipynb
│
├── data/
│   ├── raw/                           ← source CSVs (not committed)
│   ├── enriched/                      ← Gemini-enriched item data
│   ├── candidate_generation/          ← rank_train.csv / rank_test.csv
│   └── model_input/                   ← lgbm_train/test_features_v2.csv
│
├── src/
│   ├── data_loader.py                 ← load & validate input data
│   ├── feature_pipeline.py            ← cart-state, user, item feature builders
│   ├── retrieval_engine.py            ← co-occurrence + ctx + rule-fill retrieval
│   ├── ranking_model.py               ← LightGBM train / predict wrappers
│   ├── evaluation.py                  ← NDCG@K, Precision@K, Recall@K, MRR
│   ├── ab_testing.py                  ← offline A/B simulation + significance tests
│   └── inference_pipeline.py          ← end-to-end online inference (<200ms)
│
├── models/
│   └── csao_ranker.lgb                ← trained LightGBM binary ranker (v1)
│
├── reports/
│   ├── error_analysis.pdf             ← hits vs misses, price tier, volume bucket
│   ├── constraints_and_scalability.pdf
│   └── architecture_diagram.png
│
├── requirements.txt
└── LICENSE
```

---

## 📓 Notebooks

Run the notebooks in order on **Google Colab** (GPU not required):

### `DS_Enrichment.ipynb` — LLM Data Enrichment
Uses **Gemini 1.5 Flash** to enrich raw menu item data:
- Category normalisation across heterogeneous restaurant menus
- Multi-label `meal_time_suitability` tags (a Butter Chicken can be both lunch *and* dinner)
- Dish subtype inference (e.g., `Biryani → main`, `Gulab Jamun → dessert`)

```python
# Key design decision: meal_time_suitability is a LIST, not a single value
# Real food items span multiple meal times — modelled as multi-hot encoding
item["meal_time_suitability"] = ["lunch", "dinner"]  # not just "dinner"
```

---

### `01_split_and_feature_engineering.ipynb` — Feature Pipeline

**Input:** raw sessions, cart events, users, restaurants, items  
**Output:** `lgbm_train_features_v2.csv` (2,133,065 rows × 75 cols), `lgbm_test_features_v2.csv` (533,556 rows)

**Session-level 80/20 split** (no data leakage — full sessions go to one split):
```
Train: 2,133,065 rows | 80,916 query groups | positive rate: 3.79%
Test:    533,556 rows | 20,246 query groups | positive rate: 3.79%
```

**47 features across 5 categories:**

| Category | Features |
|---|---|
| **Retrieval Signal** | `retrieval_score`, `src_cooc`, `src_ctx`, `src_rule`, `retrieval_x_attach`, `retrieval_x_popularity` |
| **Cart Context** | `cart_item_count`, `cart_total_value`, `cart_has_{main,beverage,dessert,side}`, `missing_{bev,des,side}_flag`, `step_decay`, `last_item_category_*`, `effective_category_*` |
| **User History** | `order_count_90d`, `recency_days`, `avg_order_value`, `veg_preference_ratio`, `dessert_affinity_score`, `beverage_affinity_score`, `price_sensitivity_score` |
| **Item Features** | `price`, `popularity_score`, `historical_attach_rate`, `overall_attach_rate`, `aggregate_rating`, `meal_time_specificity`, `meal_time_overlap` |
| **Interactions** | `x_missing_bev_aff`, `x_missing_des_aff`, `x_price_sens_price`, `x_cart_value_price`, `missing_bev_x_retrieval`, `missing_des_x_retrieval`, `x_cand_bev_aff`, `x_cand_des_aff` |

**Sequential cart logic:**  
`step_decay × cart_has_* × effective_category_*` encodes which meal components are still missing at each step, directly modelling the Biryani → Salan → Gulab Jamun → Drinks progression.

---

### `02_candidate_generation.ipynb` — Retrieval Stage

**Hyperparameters:**
```python
COOC_TOP_PER_ANCHOR = 100    # top co-occurring items per anchor
CTX_TOP_PER_BUCKET  = 50     # top items per (restaurant, meal_time) bucket
N_CANDIDATES        = 30     # final candidates per (session, step)
COOC_WEIGHT         = 0.7    # blend: 70% co-occurrence
CTX_WEIGHT          = 0.3    # blend: 30% context-popularity
RULE_BOOST          = 0.05   # score boost for rule-fill candidates
```

**Three retrieval sources fused into a single ranked list:**

| Source | Method | Coverage |
|---|---|---|
| **Co-occurrence** | Weighted item-item pairs from training cart sessions | 99.96% |
| **Context popularity** | Top items per (restaurant_id, meal_time_bucket) | 100.00% |
| **Rule-fill** | Category-level heuristics (missing beverage → add beverages) | 96.44% |

**Output per (session, step) group:**
- `retrieval_score` — blended co-occurrence + context signal
- `src_cooc / src_ctx / src_rule` — binary source flags used as model features
- `label_addon_added` — next-step label (which item was actually added at step t+1)

---

### `03_model_training_lightgbm.ipynb` — Model Training & Evaluation

**Model:** LightGBM Binary Classifier (v4 format)

```python
params = {
    "objective":         "binary",
    "metric":            "auc",
    "num_leaves":        63,
    "learning_rate":     0.05,
    "scale_pos_weight":  25.36,     # neg/pos ratio in training set
    "lambda_l1":         0.0,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "feature_fraction":  0.8,
    "min_data_in_leaf":  20,
    "max_bin":           255,
    "num_boost_round":   3000,
    "early_stopping_rounds": 100,
}
```

**Model file:** `models/csao_ranker.lgb`
- Format: LightGBM text (v4), 7 trees, 63 leaves per tree
- 47 features, binary sigmoid objective
- Size: ~312 lines (compact, production-ready)

**Top Feature Importances (Gain):**

```
 1. retrieval_score          11,933,007  ████████████████████████████  78.6%
 2. historical_attach_rate      783,044  ██                             5.2%
 3. src_cooc                    579,677  █                              3.8%
 4. cart_has_beverage           522,703  █                              3.4%
 5. src_rule                    416,742  █                              2.7%
 6. effective_category_dessert  325,544                                 2.1%
 7. effective_category_side     271,826                                 1.8%
 8. popularity_score            171,156                                 1.1%
 9. cart_has_dessert            168,616                                 1.1%
10. step_decay                   99,888                                 0.7%
```

**Operational Metrics:**
```
Inference latency (25 candidates):
  P50: 0.068 ms  |  P95: 0.088 ms  |  P99: 0.119 ms  |  P999: 0.328 ms
  → Well within 200–300ms SLA (model uses <0.12% of latency budget)

Coverage:  100% of test groups receive predictions
Cold-start: 96.44% covered via rule-fill fallback
```

---

### `04_ab_testing_simulation.ipynb` — A/B Test Simulation

Full offline A/B simulation comparing **Control** (retrieval_score baseline) vs **Treatment** (LightGBM Ranker):

**Statistical Significance:**

| Metric | Control | Treatment | Lift | p-value | Cohen d | Significant? |
|---|---|---|---|---|---|---|
| NDCG@8 | 0.3987 | 0.4556 | +0.0570 | < 1e-300 | 0.336 | ✅ Yes |
| Recall@8 | 0.6389 | 0.7158 | +0.0769 | 7.2e-243 | 0.237 | ✅ Yes |
| MRR | 0.3485 | 0.3941 | +0.0456 | < 1e-300 | 0.287 | ✅ Yes |
| Hit@1 | 0.1856 | 0.2133 | +0.0277 | 7.8e-63 | 0.118 | ✅ Yes |

**Sample size analysis:** At 5M daily sessions with 2-week run, detectable MDE = 0.03% relative (our observed effect of +12% is easily detectable).

---

## 🔬 Feature Engineering Deep Dive

### Cart State Encoding (Sequential Logic)

```python
# At each step, we track meal composition and what's missing
cart_has_main      = 1 if any cart item is category "main" else 0
cart_has_beverage  = 1 if any cart item is category "beverage" else 0
missing_bev_flag   = 1 - cart_has_beverage   # incomplete meal signal

# Step decay: later steps get lower weight (first add-on matters most)
step_decay = 1.0 / step_number   # step 1=1.0, step 2=0.5, step 3=0.33 ...

# Interaction features
x_missing_bev_aff = missing_beverage_flag × beverage_affinity_score
missing_bev_x_retrieval = missing_beverage_flag × retrieval_score
```

### Cold Start Strategy

```
New User    → fall back to global popularity + restaurant best-sellers
New Item    → rule-fill via category (if item is "dessert", boost with dessert attach rates)
New Rest.   → context-popularity table built from similar restaurants (same cuisine type)
Tail Items  → src_rule flag + RULE_BOOST ensures they appear in candidate set
```

### Feature Freshness

| Feature Group | Update Frequency | Source |
|---|---|---|
| Cart state features | **Real-time** (<5ms on cart-add event) | Live cart stream |
| Co-occurrence tables | **Daily** (batch rebuild, ~7s) | Cart events DWH |
| Context popularity | **Daily** (per restaurant × meal_time) | Cart events DWH |
| User history (90d) | **Daily** (order_count, recency, AOV) | Orders DWH |
| Item features | **Weekly** (price, popularity, ratings) | Menu onboarding |
| Model | **Weekly** (full pipeline <2 min) | Retrain job |

---

## ⚡ Quickstart

### Prerequisites

```bash
git clone https://github.com/your-org/CartNext-Recommendation-System.git
cd CartNext-Recommendation-System
pip install -r requirements.txt
```

### requirements.txt

```
lightgbm>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
google-generativeai>=0.3.0   # for DS_Enrichment notebook
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Run Inference on Saved Model

```python
import lightgbm as lgb
import pandas as pd
import numpy as np

# Load model
model = lgb.Booster(model_file="models/csao_ranker.lgb")

# Feature columns (must match training order)
FEATURE_COLS = [
    "retrieval_score", "src_cooc", "src_ctx", "src_rule", "is_weekend",
    "cart_item_count", "cart_total_value", "cart_has_main", "cart_has_beverage",
    "cart_has_dessert", "cart_has_side", "missing_beverage_flag",
    "missing_dessert_flag", "missing_side_flag", "price", "popularity_score",
    "historical_attach_rate", "order_count_90d", "recency_days", "avg_order_value",
    "veg_preference_ratio", "dessert_affinity_score", "beverage_affinity_score",
    "price_sensitivity_score", "aggregate_rating", "overall_attach_rate",
    "meal_time_specificity", "meal_time_overlap", "step_decay",
    "x_missing_bev_aff", "x_missing_des_aff", "x_price_sens_price",
    "x_cart_value_price", "retrieval_x_attach", "retrieval_x_popularity",
    "missing_bev_x_retrieval", "missing_des_x_retrieval",
    "last_item_category_beverage", "last_item_category_dessert",
    "last_item_category_main", "last_item_category_side",
    "effective_category_beverage", "effective_category_dessert",
    "effective_category_main", "effective_category_side",
    "x_cand_bev_aff", "x_cand_des_aff"
]

def rank_candidates(candidates_df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """
    Given a DataFrame of candidate items for one (session, step),
    return the top-k ranked by model score.
    """
    scores = model.predict(candidates_df[FEATURE_COLS])
    candidates_df = candidates_df.copy()
    candidates_df["model_score"] = scores
    return (candidates_df
            .sort_values("model_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True))

# Example usage
candidates = pd.read_csv("data/model_input/sample_candidates.csv")
rail = rank_candidates(candidates, top_k=8)
print(rail[["candidate_item_id", "model_score"]].to_string())
```

### Run Full Evaluation

```python
from src.evaluation import compute_grouped_metrics

# Load test features
test_df = pd.read_csv("data/model_input/lgbm_test_features_v2.csv")
test_df["model_score"] = model.predict(test_df[FEATURE_COLS])

# Evaluate
metrics = compute_grouped_metrics(
    df        = test_df,
    score_col = "model_score",
    label_col = "label_addon_added",
    group_col = "_group_key",
    ks        = (5, 8, 10)
)

for k, m in metrics.items():
    print(f"@{k}  NDCG={m['ndcg']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
```

---

## 🧪 A/B Testing

### Online Experiment Design

```
Hypothesis   : LightGBM ranker increases CSAO attach rate vs retrieval baseline by ≥1.5pp
Control (A)  : Stage-1 retrieval_score ranking (no ML ranker)
Treatment (B): LightGBM Binary v1 ranker on top of Stage-1 candidates

Randomisation: user_id (user-level, avoids SUTVA violation)
Traffic split: 50% / 50%
Hash function: MD5(user_id + experiment_id) % 100 < 50 → Control
Duration     : 14–21 days (2 full weekly cycles, novelty effect window)
Ramp         : Day 1 at 5% → Day 2 full traffic (SLO sanity check)
```

### Primary Metrics

| Metric | Definition | Success Threshold |
|---|---|---|
| CSAO Attach Rate | Sessions with ≥1 rail item added / sessions with rail shown | +1.5pp absolute (stat. sig.) |
| AOV Lift | Avg order value: treatment vs control | +2% relative |
| Rail Attach Rate | % orders with CSAO contribution | +1pp |
| Avg Items/Order | Cart depth increase | +0.1 items |

### Guardrail Metrics (must not degrade)

| Guardrail | Threshold | Rationale |
|---|---|---|
| Cart-to-Order ratio | < −0.5pp drop | Rail must not disrupt checkout flow |
| Order cancellation rate | No increase | No UX confusion |
| Session duration | < +2s | Rail latency must be invisible |
| Post-order rating | No significant drop | User satisfaction proxy |

### Offline → Online Translation

| Offline Metric | Online Proxy | Caution |
|---|---|---|
| **NDCG@8 = 0.4556** | CTR / Click-through rate | NDCG assumes uniform relevance; online has position bias |
| **Recall@8 = 71.58%** | Acceptance Rate / Attach Rate | Offline uses next-step label; online includes browse behaviour |
| **MRR = 0.3941** | Position-weighted CTR | Rail position primacy differs offline vs online |
| **Hit@1 = 21.33%** | Top-slot CTR | Cold-start users may behave differently than training data |

---

## 📦 Data Schema

### Input: Cart Event

```json
{
  "session_id": 12345,
  "step_number": 2,
  "user_id": 67890,
  "restaurant_id": 111,
  "item_id": 772,
  "timestamp": "2024-01-15T13:24:00Z",
  "meal_time_bucket": "lunch"
}
```

### Output: Ranked Recommendations

```json
{
  "session_id": 12345,
  "step_number": 2,
  "recommendations": [
    {"item_id": 815, "score": 0.3829, "rank": 1},
    {"item_id": 761, "score": 0.2981, "rank": 2},
    {"item_id": 742, "score": 0.2744, "rank": 3}
  ],
  "inference_latency_ms": 0.47,
  "retrieval_sources": ["cooc", "ctx"]
}
```

---

## 🤖 AI Edge — Gemini Enrichment

The `DS_Enrichment.ipynb` notebook uses **Gemini 1.5 Flash** to solve three hard data problems that traditional ML cannot address:

**1. Category Normalisation**  
Menu items arrive with inconsistent, restaurant-specific names. Gemini normalises them to a canonical taxonomy (`main`, `beverage`, `dessert`, `side`, `snack`).

**2. Multi-label Meal Time Tags**  
Unlike a naive single-label approach, Gemini tags each item with *all* applicable meal times:
```
Butter Chicken   → ["lunch", "dinner"]
Masala Chai      → ["breakfast", "lunch", "late-night"]
Chicken Biryani  → ["lunch", "dinner", "late-night"]
```
This multi-hot encoding powers the `meal_time_specificity` and `meal_time_overlap` features.

**3. Dish Subtype Inference**  
Gemini infers fine-grained dish types (e.g., `rice_dish`, `curry`, `bread`, `fried_snack`) used to build cross-category completion heuristics for the rule-fill retrieval source.

---

## 📈 Notebooks Execution Order

```
DS_Enrichment.ipynb              ← LLM enrichment (Gemini API key required)
        ↓
01_split_and_feature_engineering.ipynb   ← data split + 47 features
        ↓
02_candidate_generation.ipynb    ← retrieval stage (co-occ + ctx + rule)
        ↓
03_model_training_lightgbm.ipynb ← train, evaluate, save model
        ↓
04_ab_testing_simulation.ipynb   ← offline A/B + business projections
```

---

## 👥 Team TrailBlazers

**This project was developed by:**

Avani Agnihotri

Nehal Aggarwal

Nandini Goel



**Core stack:**

LightGBM

Python

Google Colab

---

<div align="center">

**Built with ❤️ for smarter food delivery experiences**

*LightGBM · Gemini 1.5 Flash · Google Colab · Python*

</div>
