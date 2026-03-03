# Candidate Generation (Stage 1 Retrieval)

This folder contains intermediate artifacts produced during the
candidate retrieval stage of the two-stage recommendation system.

Goal:
Given a cart state at step N, generate a high-recall set of candidate
items likely to be added at the next step (N+1).

This stage focuses on recall and coverage.
Final ranking is handled in Stage 2 using LightGBM.

---

## Overview of Retrieval Strategy

We use a hybrid retrieval approach combining:

1. Item-to-item co-occurrence
2. Context-aware popularity
3. Rule-based complementary category fill

This ensures:
- High recall
- Business-aware diversity
- Cold-start robustness
- Restaurant-level constraints

---

## Label Definition (Critical Design Choice)

We use a **next-step-only label**.

For a cart at step N:
- The positive label is ONLY the item added at step N+1.
- We do NOT consider all future items in the session.

Why?

Predicting "any future item in the session" introduces noise because
later cart states depend on many unknown interactions.

Predicting the immediate next addition:
- Is actionable
- Aligns with real-time UI recommendation
- Ensures each query group has exactly 1 positive
- Eliminates zero-positive training groups

We also exclude the last cart step during candidate generation,
because it has no next item and would always produce zero-positive groups.

---

## Files in This Folder

### cooc_top_sample.csv

Top-K item co-occurrence candidates per (restaurant_id, anchor_item_id).

Built from training sessions only.

Construction:
- For each session, accumulate weighted co-occurrence pairs.
- Weight = 1 / (1 + step_distance)
- Keep top COOC_TOP_PER_ANCHOR per anchor item.

Purpose:
Capture strong sequential purchase patterns within a restaurant.

---

### ctx_pop_top_sample.csv

Top-K context-popularity candidates per (restaurant_id, meal_time_bucket).

Construction:
- Count item frequency per restaurant and meal_time_bucket.
- Normalize within bucket.
- Keep top CTX_TOP_PER_BUCKET items.

Purpose:
Model meal-time-specific preferences (e.g., beverages at dinner).

---

### train_sessions_sample.csv
### test_sessions_sample.csv

Session-level splits used to:
- Prevent leakage in co-occurrence computation
- Build retrieval tables from training sessions only
- Evaluate ranking on held-out sessions

---

## Candidate Pool Construction

For each (session_id, step_number):

1. Define cart_so_far = items added up to current step.
2. Generate candidates from:
   - Co-occurrence lookup (anchor-based)
   - Context popularity (meal-time bucket)
   - Rule-fill for missing complementary categories
3. Apply hard constraints:
   - Exclude items already in cart
   - Restrict to same restaurant
4. Ensure the true next_item is always present in the candidate pool.
5. Truncate to N_CANDIDATES (≈ menu size).

Final output:
~25–30 candidates per cart state.

---

## Design Rationale

Why hybrid retrieval?

- Co-occurrence captures behavioral affinity.
- Context popularity captures situational demand.
- Rule-fill enforces business logic (e.g., suggest dessert if missing).
- Hard restaurant constraint ensures realistic UI behavior.

This design balances:
- Personalization
- Context awareness
- Business objectives
- System scalability

---

## Production Considerations

- Retrieval is precomputed from training sessions.
- Lookup dictionaries enable low-latency inference.
- Candidate pool size is bounded (<30).
- Compatible with real-time inference budgets (<50ms).

---

This stage maximizes Recall@K and feeds into the Stage 2 ranking model.
