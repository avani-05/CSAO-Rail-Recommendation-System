"""
src/evaluation.py
==================
Offline evaluation metrics for the CSAO Rail Recommendation System.

All metrics are computed at the query-group level and averaged across groups.
A query group is one (session_id, step_number) pair — i.e., one cart state
presenting ~25 candidates.

Metrics
-------
  NDCG@K       — Normalised Discounted Cumulative Gain
  Precision@K  — fraction of top-K recommendations that are relevant
  Recall@K     — fraction of relevant items found in top-K
  Hit@K        — binary: was any relevant item in the top-K?
  MRR          — Mean Reciprocal Rank (rank of first relevant item)
  Hit@1        — was the top-1 item correct?

Usage
-----
    from src.evaluation import compute_metrics, compare_models, print_report

    # Single model
    metrics = compute_metrics(test_df, score_col="model_score",
                              label_col="label_addon_added",
                              group_col="_group_key", ks=(5, 8, 10))

    # Model vs baselines
    report = compare_models(test_df, {
        "LightGBM": "model_score",
        "Retrieval": "retrieval_score",
        "Popularity": "popularity_score",
    })
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Core DCG / NDCG ─────────────────────────────────────────────────────────────

def dcg_at_k(relevance: List[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    r = np.array(relevance[:k], dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))


def ndcg_at_k(labels_sorted_by_score: List[int], k: int) -> float:
    """Normalised DCG@K (ideal DCG = perfect ranking)."""
    actual = dcg_at_k(labels_sorted_by_score, k)
    ideal  = dcg_at_k(sorted(labels_sorted_by_score, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


# ── Per-group metric computation ─────────────────────────────────────────────────

def _group_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    ks:     Tuple[int, ...],
) -> Optional[dict]:
    """
    Compute metrics for a single query group.
    Returns None if the group has no positive labels.
    """
    n_pos = labels.sum()
    if n_pos == 0:
        return None

    n          = len(labels)
    ranked_idx = np.argsort(scores)[::-1]
    ranked_lbl = labels[ranked_idx].tolist()

    rec = {"n_pos": int(n_pos)}

    for k in ks:
        k_eff   = min(k, n)
        top_k   = ranked_lbl[:k_eff]
        n_hits  = sum(top_k)

        rec[f"ndcg@{k}"]      = ndcg_at_k(ranked_lbl, k)
        rec[f"precision@{k}"] = n_hits / k
        rec[f"recall@{k}"]    = n_hits / n_pos
        rec[f"hit@{k}"]       = int(n_hits > 0)

    # MRR — rank of first relevant item
    rec["mrr"] = 0.0
    for rank, lbl in enumerate(ranked_lbl, 1):
        if lbl == 1:
            rec["mrr"] = 1.0 / rank
            break

    # Hit@1
    rec["hit@1"] = int(ranked_lbl[0] == 1)

    return rec


# ── Main API ──────────────────────────────────────────────────────────────────────

def compute_metrics(
    df:        pd.DataFrame,
    score_col: str  = "model_score",
    label_col: str  = "label_addon_added",
    group_col: str  = "_group_key",
    ks:        Tuple[int, ...] = (5, 8, 10),
) -> Dict[int, dict]:
    """
    Compute mean NDCG@K, Precision@K, Recall@K, Hit@K across all query groups.

    Parameters
    ----------
    df        : scored candidate DataFrame
    score_col : column with predicted scores (higher = better)
    label_col : binary relevance label column
    group_col : column identifying each query group
    ks        : list of cutoff values

    Returns
    -------
    dict  { k: {"ndcg": float, "precision": float, "recall": float,
                "hit": float, "n_groups": int} }
    """
    results = {k: {"ndcg": [], "precision": [], "recall": [], "hit": []} for k in ks}
    n_skipped = 0

    for _, grp in df.groupby(group_col, sort=False):
        labels = grp[label_col].values.astype(int)
        scores = grp[score_col].values.astype(float)
        rec    = _group_metrics(labels, scores, ks)
        if rec is None:
            n_skipped += 1
            continue
        for k in ks:
            results[k]["ndcg"].append(rec[f"ndcg@{k}"])
            results[k]["precision"].append(rec[f"precision@{k}"])
            results[k]["recall"].append(rec[f"recall@{k}"])
            results[k]["hit"].append(rec[f"hit@{k}"])

    if n_skipped > 0:
        logger.debug("Skipped %d groups with no positive labels", n_skipped)

    aggregated = {}
    for k in ks:
        n_groups = len(results[k]["ndcg"])
        aggregated[k] = {
            "ndcg":      float(np.mean(results[k]["ndcg"])),
            "precision": float(np.mean(results[k]["precision"])),
            "recall":    float(np.mean(results[k]["recall"])),
            "hit":       float(np.mean(results[k]["hit"])),
            "n_groups":  n_groups,
        }
    return aggregated


def compute_per_group_metrics(
    df:        pd.DataFrame,
    score_col: str  = "model_score",
    label_col: str  = "label_addon_added",
    group_col: str  = "_group_key",
    ks:        Tuple[int, ...] = (5, 8, 10),
) -> pd.DataFrame:
    """
    Return per-group metric DataFrame (one row per query group).
    Useful for statistical significance testing.
    """
    records = []
    for gk, grp in df.groupby(group_col):
        labels = grp[label_col].values.astype(int)
        scores = grp[score_col].values.astype(float)
        rec    = _group_metrics(labels, scores, ks)
        if rec is None:
            continue
        rec["group_key"] = gk
        records.append(rec)
    return pd.DataFrame(records)


def compute_mrr(
    df:        pd.DataFrame,
    score_col: str = "model_score",
    label_col: str = "label_addon_added",
    group_col: str = "_group_key",
) -> float:
    """Compute Mean Reciprocal Rank."""
    rr_list = []
    for _, grp in df.groupby(group_col, sort=False):
        labels = grp[label_col].values.astype(int)
        scores = grp[score_col].values.astype(float)
        if labels.sum() == 0:
            continue
        ranked = labels[np.argsort(scores)[::-1]]
        for rank, lbl in enumerate(ranked, 1):
            if lbl == 1:
                rr_list.append(1.0 / rank)
                break
    return float(np.mean(rr_list)) if rr_list else 0.0


def compute_hit_rate(
    df:        pd.DataFrame,
    score_col: str = "model_score",
    label_col: str = "label_addon_added",
    group_col: str = "_group_key",
    k:         int = 8,
) -> float:
    """Compute Hit Rate @K (fraction of groups where relevant item is in top-K)."""
    hits = total = 0
    for _, grp in df.groupby(group_col, sort=False):
        labels = grp[label_col].values.astype(int)
        scores = grp[score_col].values.astype(float)
        if labels.sum() == 0:
            continue
        ranked = labels[np.argsort(scores)[::-1]]
        hits  += int(ranked[:k].sum() > 0)
        total += 1
    return hits / total if total > 0 else 0.0


# ── Multi-model comparison ────────────────────────────────────────────────────────

def compare_models(
    df:         pd.DataFrame,
    models:     Dict[str, str],
    label_col:  str  = "label_addon_added",
    group_col:  str  = "_group_key",
    ks:         Tuple[int, ...] = (5, 8, 10),
) -> pd.DataFrame:
    """
    Compute and compare metrics for multiple models / baselines.

    Parameters
    ----------
    df     : DataFrame containing score columns for all models
    models : dict mapping model_name → score_column_name
             e.g. {"LightGBM": "model_score", "Retrieval": "retrieval_score"}

    Returns
    -------
    pd.DataFrame with one row per (model, metric@K)
    """
    rows = []
    for model_name, score_col in models.items():
        if score_col not in df.columns:
            logger.warning("Score column '%s' not found — skipping %s", score_col, model_name)
            continue
        metrics = compute_metrics(df, score_col, label_col, group_col, ks)
        for k, m in metrics.items():
            rows.append({
                "model":     model_name,
                "k":         k,
                "ndcg":      m["ndcg"],
                "precision": m["precision"],
                "recall":    m["recall"],
                "hit":       m["hit"],
                "n_groups":  m["n_groups"],
            })
    return pd.DataFrame(rows)


def significance_test(
    df:           pd.DataFrame,
    score_a:      str,
    score_b:      str,
    label_col:    str  = "label_addon_added",
    group_col:    str  = "_group_key",
    key_metrics:  Tuple[str, ...] = ("ndcg@8", "recall@8", "mrr", "hit@1"),
    ks:           Tuple[int, ...] = (8,),
    alpha:        float = 0.05,
) -> pd.DataFrame:
    """
    Paired t-test + Wilcoxon signed-rank test between two scoring systems.

    Returns
    -------
    pd.DataFrame with columns:
        metric, ctrl_mean, treat_mean, mean_diff, ci_low, ci_high,
        p_val_t, p_val_w, cohen_d, significant
    """
    from scipy.stats import ttest_rel, wilcoxon

    grp_a = compute_per_group_metrics(df, score_a, label_col, group_col, ks)
    grp_b = compute_per_group_metrics(df, score_b, label_col, group_col, ks)

    merged = grp_a.set_index("group_key").join(
        grp_b.set_index("group_key"), lsuffix="_a", rsuffix="_b"
    ).dropna()

    rows = []
    for m in key_metrics:
        col_a = f"{m}_a" if f"{m}_a" in merged.columns else m
        col_b = f"{m}_b" if f"{m}_b" in merged.columns else m
        if col_a not in merged.columns or col_b not in merged.columns:
            continue

        va = merged[col_a].values
        vb = merged[col_b].values
        d  = vb - va
        n  = len(d)
        se = d.std() / np.sqrt(n)

        t_stat, p_t = ttest_rel(vb, va)
        try:
            _, p_w = wilcoxon(vb, va)
        except ValueError:
            p_w = 1.0

        rows.append({
            "metric":     m,
            "ctrl_mean":  float(va.mean()),
            "treat_mean": float(vb.mean()),
            "mean_diff":  float(d.mean()),
            "ci_low":     float(d.mean() - 1.96 * se),
            "ci_high":    float(d.mean() + 1.96 * se),
            "p_val_t":    float(p_t),
            "p_val_w":    float(p_w),
            "cohen_d":    float(d.mean() / (d.std() + 1e-9)),
            "significant": p_t < alpha,
            "n_groups":   int(n),
        })
    return pd.DataFrame(rows)


# ── Segment-level breakdown ──────────────────────────────────────────────────────

def segment_breakdown(
    df:          pd.DataFrame,
    score_col:   str,
    segment_col: str,
    label_col:   str = "label_addon_added",
    group_col:   str = "_group_key",
    k:           int = 8,
) -> pd.DataFrame:
    """
    Compute NDCG@K and Recall@K broken down by a segment column.

    Parameters
    ----------
    df          : scored DataFrame (must contain segment_col)
    score_col   : score column name
    segment_col : column to segment by (e.g. "user_segment", "meal_time_bucket")

    Returns
    -------
    pd.DataFrame with columns: segment, n_groups, ndcg@K, recall@K, hit@K
    """
    rows = []
    for seg_val, seg_grp in df.groupby(segment_col):
        records = []
        for _, grp in seg_grp.groupby(group_col):
            labels = grp[label_col].values.astype(int)
            scores = grp[score_col].values.astype(float)
            rec    = _group_metrics(labels, scores, (k,))
            if rec is None:
                continue
            records.append(rec)
        if not records:
            continue
        sdf = pd.DataFrame(records)
        rows.append({
            "segment":      seg_val,
            "n_groups":     len(sdf),
            f"ndcg@{k}":   float(sdf[f"ndcg@{k}"].mean()),
            f"recall@{k}": float(sdf[f"recall@{k}"].mean()),
            f"hit@{k}":    float(sdf[f"hit@{k}"].mean()),
        })
    return pd.DataFrame(rows).sort_values(f"ndcg@{k}", ascending=False)


# ── Pretty-print report ──────────────────────────────────────────────────────────

def print_report(
    metrics_dict: Dict[int, dict],
    model_name:   str = "Model",
) -> None:
    """Print a formatted evaluation table."""
    print(f"\n{'='*56}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*56}")
    print(f"  {'Metric':<20} {'@5':>8} {'@8':>8} {'@10':>8}")
    print(f"  {'-'*44}")
    for metric in ("ndcg", "precision", "recall", "hit"):
        vals = [metrics_dict.get(k, {}).get(metric, 0.0) for k in (5, 8, 10)]
        print(f"  {metric.upper():<20} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")
    print(f"{'='*56}")
    if 8 in metrics_dict:
        print(f"  Groups evaluated : {metrics_dict[8]['n_groups']:,}")
    print()
