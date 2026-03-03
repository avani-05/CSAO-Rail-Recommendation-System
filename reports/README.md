# Reports & Documentation

This folder contains detailed analysis, system design documentation,
and the final consolidated project report for the two-stage
cart-based recommendation system.

These artifacts complement the notebooks and modular `src/` code
by documenting modeling decisions, trade-offs, and business impact.

---

## 📘 Cart_Next_CSAO_Recommendation_Report.pdf

Final consolidated project report.

This document provides an end-to-end overview of:

- Problem framing (neutral description)
- Data pipeline design
- Two-stage system architecture
- Candidate generation strategy
- Feature engineering methodology
- Model training & evaluation
- Offline A/B simulation results
- Business impact translation
- Scalability considerations
- Future improvements

This is the primary document for understanding the complete system.

---

## 📊 CSAO_AB_Test_Report.pdf

Offline A/B simulation comparing:

- Retrieval-only ranking
- Learned LightGBM ranking model

Includes:

- NDCG@K comparison
- Recall@K uplift
- Score distribution analysis
- Attach-rate proxy lift
- Visibility improvement estimation

Purpose:
Translate model performance into measurable business impact.

---

## ⚙️ CSAO_Constraints_Report.pdf

System-level design constraints and trade-offs:

- Latency budget reasoning
- Candidate pool size limits
- Real-time inference considerations
- Memory footprint
- Restaurant-level filtering logic
- Cold-start handling strategy

Purpose:
Demonstrate production readiness and scalability awareness.

---

## 🔍 csao_error_analysis_report.pdf

Detailed analysis of ranking errors:

- Missed positives
- False positives
- Category imbalance
- Popularity bias effects
- Cart-state edge cases

Includes qualitative and quantitative breakdown of model weaknesses
to guide future improvements.

---

## 🧠 model_arch.png

High-level architecture diagram of the two-stage system.

Stage 1: Candidate Retrieval  
- Item co-occurrence  
- Context popularity (meal-time buckets)  
- Rule-based complementary fills  

Stage 2: Ranking Model  
- Feature engineering  
- LightGBM scoring  
- Top-K selection  

Also highlights:

- Data flow across stages
- Candidate bounding
- Inference pipeline structure
- Latency-aware design

---

## Why This Folder Matters

Machine learning systems are not just about model accuracy.

This folder demonstrates:

- End-to-end system thinking
- Structured experimentation
- Business translation of metrics
- Production-aware architecture
- Clear documentation of trade-offs

Together, these reports position the project as a complete,
real-world recommendation system rather than a standalone model experiment.
