
# Data Pipeline Overview

This folder contains all datasets used in the cart-based
next-item recommendation system.

The structure reflects the full data lifecycle:

Raw → Enriched → Candidate Generation → Model Input

Each stage builds upon the previous one and produces
artifacts consumed by the next stage.

Only small representative samples are included in this repository
to ensure reproducibility without exposing large or sensitive data.

---

## Folder Structure

data/
├── Raw/
├── Enriched/
├── Candidate_Generation/
└── model_input/

---

## 1️⃣ Raw/

Original structured event-level data.

Includes sample versions of:

- users
- sessions
- restaurants
- cart_events
- items_noisy

This layer represents the initial input before any cleaning,
enrichment, or feature engineering.

---

## 2️⃣ Enriched/

Contains transformed datasets after:

- Noise removal from item metadata
- LLM-based menu normalization
- Category standardization
- Session cleaning
- Cart state preparation

Key outputs:
- items_llm_tags
- cleaned cart events
- cleaned sessions

Purpose:
Improve metadata quality and feature reliability.

---

## 3️⃣ Candidate_Generation/

Stage 1 of the recommendation system.

Artifacts generated here:

- Item-to-item co-occurrence tables
- Context-aware popularity tables
- Train/test session splits
- Candidate-expanded cart states

Each cart state is expanded into a bounded candidate pool
(~25–30 items) for ranking.

This stage prioritizes high recall.

---

## 4️⃣ model_input/

Stage 2 training data.

Contains:

- Candidate-level labeled datasets
- Final feature matrices for LightGBM
- Train/test splits aligned by session

Each row represents:
(session_id, step_number, candidate_item_id)

Exactly one positive per group (next-step-only label).

These files are directly used to train the ranking model.

---

## Label Definition

We use a strict next-step-only label:

For a cart at step N,
the positive example is the item added at step N+1.

This ensures:
- Exactly one positive per group
- Clean ranking structure
- No future leakage

---

## Reproducibility Notes

- All transformations are implemented in `notebooks/`
  and modularized in `src/`.
- Only small sample datasets are included here.
- Full datasets are not uploaded to keep the repository lightweight.

---

## Why This Structure Matters

Organizing data by pipeline stage:

- Improves clarity
- Prevents data leakage
- Supports reproducibility
- Reflects production-grade system design

This is not just a dataset dump —
it mirrors the full recommendation system lifecycle.
