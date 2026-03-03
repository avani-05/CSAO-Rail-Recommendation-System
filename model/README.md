
# Trained Model Artifacts

This folder contains the trained ranking model used
for next-item recommendation.

---

## 📦 csao_ranker.lgb

LightGBM binary classification model trained on the
candidate-expanded ranking dataset.

Objective:
Predict the probability that a candidate item
is the next item added to the cart.

The predicted probability is used as the ranking score.

---

## Model Details

Model type: LightGBM (Gradient Boosted Decision Trees)  
Objective: Binary classification  
Evaluation metrics: AUC, NDCG@8, Recall@8  

Training data:
- Generated from `data/model_input/`
- Grouped by (session_id, step_number)
- Exactly 1 positive per group

Feature set:
- Retrieval signals
- Cart state features
- User behavioral features
- Item attributes
- Cross features

Final feature count: ~47 features

---

## How to Load the Model

Example usage:

```python
from src.ranking_model import CartRanker

ranker = CartRanker()
ranker.load("models/csao_ranker.lgb")

scores = ranker.predict(feature_matrix)
