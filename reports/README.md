
# Reports & Analysis

This folder contains supporting analysis, system design documentation,
and business impact evaluation for the cart-based recommendation system.

These reports complement the notebooks and modular `src/` code by
providing deeper reasoning behind modeling decisions.

---

## 📊 CSAO_AB_Test_Report.pdf

Offline A/B simulation results comparing:

- Baseline retrieval ranking
- Learned LightGBM ranking model

Includes:

- NDCG@K comparison
- Recall@K uplift
- Score distribution analysis
- Attach-rate proxy lift
- Visibility improvement estimation

Purpose:
Translate ranking improvements into measurable business impact.

---

## ⚙️ CSAO_Constraints_Report.pdf

Details system-level constraints and design trade-offs:

- Latency budget reasoning
- Candidate pool size limits
- Real-time inference considerations
- Memory constraints
- Restaurant-level filtering logic
- Cold-start handling strategy

Purpose:
Demonstrate production readiness and scalability awareness.

---

## 🔍 csao_error_analysis_report.pdf

Focused error breakdown of ranking failures:

- Missed positives analysis
- False positive patterns
- Category imbalance effects
- Popularity bias evaluation
- Cart-state edge cases

Includes examples of:

- High retrieval score but wrong ranking
- Cold-start misclassifications
- Context misalignment

Purpose:
Identify model weaknesses and guide future improvements.

---

## 🧠 model_arch.png

High-level architecture diagram of the two-stage system:

Stage 1: Candidate Retrieval
- Item co-occurrence
- Context popularity (meal-time buckets)
- Rule-based complementary fills

Stage 2: Ranking Model
- Feature engineering
- LightGBM scoring
- Top-K selection

Also highlights:

- Latency budget
- Candidate size constraints
- Data flow between components

---

## Why These Reports Matter

Machine learning systems are not just about model metrics.

This folder demonstrates:

- System design clarity
- Trade-off reasoning
- Business translation
- Analytical rigor
- Production awareness

Together, these artifacts show the project as a
complete, end-to-end recommendation system rather
than a standalone model experiment.
