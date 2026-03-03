"""
src — CartNext CSAO Rail Recommendation System
===============================================

Modules
-------
data_loader        Load and validate raw CSV input data
feature_pipeline   Build the 47-feature matrix from raw tables
retrieval_engine   Stage-1: candidate generation (co-occurrence + ctx + rule)
ranking_model      Stage-2: LightGBM binary ranker (train / predict / save / load)
evaluation         Offline metrics: NDCG@K, Precision@K, Recall@K, MRR, Hit@K
ab_testing         Offline A/B simulation, power analysis, online experiment design
inference_pipeline End-to-end online inference pipeline (<200ms)
"""

from src.data_loader        import CSAODataLoader, CSAODataset, load_dataset, load_feature_tables
from src.feature_pipeline   import FeaturePipeline
from src.retrieval_engine   import RetrievalEngine
from src.ranking_model      import CSAORanker
from src.evaluation         import (
    compute_metrics,
    compute_per_group_metrics,
    compute_mrr,
    compute_hit_rate,
    compare_models,
    significance_test,
    segment_breakdown,
    print_report,
)
from src.ab_testing         import ABTestSimulator, ExperimentDesign, GuardrailConfig, check_guardrails
from src.inference_pipeline import InferencePipeline, CartEvent, PredictionResult, Recommendation

__all__ = [
    # data
    "CSAODataLoader", "CSAODataset", "load_dataset", "load_feature_tables",
    # features
    "FeaturePipeline",
    # retrieval
    "RetrievalEngine",
    # model
    "CSAORanker",
    # evaluation
    "compute_metrics", "compute_per_group_metrics", "compute_mrr",
    "compute_hit_rate", "compare_models", "significance_test",
    "segment_breakdown", "print_report",
    # ab testing
    "ABTestSimulator", "ExperimentDesign", "GuardrailConfig", "check_guardrails",
    # inference
    "InferencePipeline", "CartEvent", "PredictionResult", "Recommendation",
]
