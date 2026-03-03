"""
src/ab_testing.py
==================
Offline A/B testing simulation and online experiment design for the
CSAO Rail Recommendation System.

Mirrors the logic in notebooks/04_ab_testing_simulation.ipynb.

Capabilities
------------
1. Offline A/B comparison   — per-group metric delta, paired t-test, Wilcoxon
2. Business metric projection — AOV lift, daily revenue, acceptance rate
3. Sample size analysis      — power analysis for online experiment design
4. Segment-level comparison  — meal_time, user_segment, step_number
5. Rollback trigger check    — evaluate guardrail metrics

Usage
-----
    from src.ab_testing import ABTestSimulator, ExperimentDesign

    sim = ABTestSimulator(test_df, model=ranker)
    sim.run(control_col="retrieval_score", treatment_col="model_score")
    sim.print_report()

    design = ExperimentDesign(ctrl_attach_rate=0.639)
    design.print_sample_size_table()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_rel, wilcoxon

from src.evaluation import (
    compute_per_group_metrics,
    segment_breakdown,
)

logger = logging.getLogger(__name__)

# ── Business constants (match notebook Cell 5) ──────────────────────────────────
DEFAULT_BIZ_CONFIG = {
    "avg_item_price_inr":     120.0,
    "baseline_avg_order_inr": 450.0,
    "daily_sessions":         5_000_000,
    "display_k":              8,
    "baseline_c2o_rate":      0.80,
    "c2o_uplift_per_hit":     0.05,
}


# ── Offline A/B Simulator ────────────────────────────────────────────────────────

class ABTestSimulator:
    """
    Simulate an offline A/B test comparing control vs treatment scoring.

    Parameters
    ----------
    df          : scored test DataFrame (must have group_col, label_col,
                  and both score columns)
    label_col   : binary relevance label
    group_col   : query group identifier
    ks          : cutoff values for ranking metrics
    biz_config  : business metric constants (override DEFAULT_BIZ_CONFIG)
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        label_col:  str   = "label_addon_added",
        group_col:  str   = "_group_key",
        ks:         Tuple[int, ...] = (5, 8, 10),
        biz_config: Optional[dict]  = None,
    ):
        self.df        = df.copy()
        self.label_col = label_col
        self.group_col = group_col
        self.ks        = ks
        self.biz       = {**DEFAULT_BIZ_CONFIG, **(biz_config or {})}

        # Populated after run()
        self.ctrl_grp:  Optional[pd.DataFrame] = None
        self.treat_grp: Optional[pd.DataFrame] = None
        self.sig_df:    Optional[pd.DataFrame] = None
        self.biz_df:    Optional[pd.DataFrame] = None

    # ── public ─────────────────────────────────────────────────────────────────

    def run(
        self,
        control_col:   str = "retrieval_score",
        treatment_col: str = "model_score",
        alpha:         float = 0.05,
    ) -> "ABTestSimulator":
        """
        Run the full A/B simulation pipeline.

        Steps:
          1. Per-group metrics for control and treatment
          2. Statistical significance tests
          3. Business metric projections

        Returns
        -------
        self (for chaining)
        """
        logger.info("Running offline A/B simulation ...")

        # 1. Per-group metrics
        logger.info("Computing control metrics (%s) ...", control_col)
        self.ctrl_grp = compute_per_group_metrics(
            self.df, control_col, self.label_col, self.group_col, self.ks
        )

        logger.info("Computing treatment metrics (%s) ...", treatment_col)
        self.treat_grp = compute_per_group_metrics(
            self.df, treatment_col, self.label_col, self.group_col, self.ks
        )

        # 2. Significance tests
        self.sig_df = self._significance_tests(alpha)

        # 3. Business projections
        self.biz_df = self._business_projections()

        logger.info("A/B simulation complete.")
        return self

    def metric_comparison(self) -> pd.DataFrame:
        """Return a summary DataFrame of control vs treatment metrics."""
        self._require_run()
        rows = []
        key_cols = (
            [f"ndcg@{k}" for k in self.ks]
            + [f"precision@{k}" for k in self.ks]
            + [f"recall@{k}" for k in self.ks]
            + [f"hit@{k}" for k in self.ks]
            + ["mrr", "hit@1"]
        )
        for col in key_cols:
            if col not in self.ctrl_grp.columns:
                continue
            c = self.ctrl_grp[col].mean()
            t = self.treat_grp[col].mean()
            rows.append({
                "metric":    col,
                "control":   round(c, 4),
                "treatment": round(t, 4),
                "lift_abs":  round(t - c, 4),
                "lift_pct":  f"{(t - c) / (c + 1e-9) * 100:+.1f}%",
            })
        return pd.DataFrame(rows)

    def segment_comparison(
        self,
        segment_col: str,
        k: int = 8,
    ) -> pd.DataFrame:
        """Compare control vs treatment broken down by a segment column."""
        self._require_run()
        if segment_col not in self.df.columns:
            raise ValueError(f"Column '{segment_col}' not found in DataFrame.")

        ctrl  = segment_breakdown(self.df, "control_score",   segment_col,
                                  self.label_col, self.group_col, k)
        treat = segment_breakdown(self.df, "model_score", segment_col,
                                  self.label_col, self.group_col, k)

        comp = ctrl.merge(treat, on="segment", suffixes=("_ctrl", "_treat"))
        comp[f"ndcg@{k}_lift"]   = (
            (comp[f"ndcg@{k}_treat"]   - comp[f"ndcg@{k}_ctrl"])
            / comp[f"ndcg@{k}_ctrl"] * 100
        ).map("{:+.1f}%".format)
        comp[f"recall@{k}_lift"] = (
            (comp[f"recall@{k}_treat"] - comp[f"recall@{k}_ctrl"])
            / comp[f"recall@{k}_ctrl"] * 100
        ).map("{:+.1f}%".format)
        return comp

    def print_report(self) -> None:
        """Print a comprehensive A/B test report."""
        self._require_run()
        n = len(self.ctrl_grp)

        print(f"\n{'='*72}")
        print("  CSAO OFFLINE A/B TEST REPORT")
        print(f"{'='*72}")
        print(f"  Groups evaluated: {n:,}")
        print()

        # Metric comparison
        print("  Metric Comparison (Control vs Treatment)")
        print(f"  {'-'*60}")
        comp = self.metric_comparison()
        for _, row in comp[comp["metric"].str.contains("@8|mrr|hit@1")].iterrows():
            print(f"  {row['metric']:<20} ctrl={row['control']:.4f}  "
                  f"treat={row['treatment']:.4f}  {row['lift_pct']}")

        # Significance
        print(f"\n  Statistical Significance (α=0.05)")
        print(f"  {'-'*60}")
        if self.sig_df is not None:
            for _, row in self.sig_df.iterrows():
                sig = "✅" if row["significant"] else "❌"
                print(
                    f"  {row['metric']:<12}  diff={row['mean_diff']:+.4f}"
                    f"  p={row['p_val_t']:.2e}  Cohen d={row['cohen_d']:.3f}  {sig}"
                )

        # Business projections
        print(f"\n  Business Impact Projections")
        print(f"  {'-'*60}")
        if self.biz_df is not None:
            for _, row in self.biz_df.iterrows():
                print(f"  {row['metric']:<40} {row['control']:>14} → {row['treatment']:<14}  {row['delta']}")
        print(f"{'='*72}\n")

    # ── private ─────────────────────────────────────────────────────────────────

    def _significance_tests(self, alpha: float) -> pd.DataFrame:
        merged = (
            self.ctrl_grp.set_index("group_key")
            .join(self.treat_grp.set_index("group_key"),
                  lsuffix="_ctrl", rsuffix="_treat")
            .dropna()
        )

        rows = []
        for m in ("ndcg@8", "recall@8", "mrr", "hit@1"):
            ca = f"{m}_ctrl"
            cb = f"{m}_treat"
            if ca not in merged.columns or cb not in merged.columns:
                continue
            va = merged[ca].values
            vb = merged[cb].values
            d  = vb - va
            n  = len(d)
            se = d.std() / np.sqrt(n)

            _, p_t = ttest_rel(vb, va)
            try:
                _, p_w = wilcoxon(vb, va)
            except ValueError:
                p_w = 1.0

            rows.append({
                "metric":      m,
                "ctrl_mean":   float(va.mean()),
                "treat_mean":  float(vb.mean()),
                "mean_diff":   float(d.mean()),
                "ci_low":      float(d.mean() - 1.96 * se),
                "ci_high":     float(d.mean() + 1.96 * se),
                "p_val_t":     float(p_t),
                "p_val_w":     float(p_w),
                "cohen_d":     float(d.mean() / (d.std() + 1e-9)),
                "significant": bool(p_t < alpha),
                "n_groups":    int(n),
            })
        return pd.DataFrame(rows)

    def _business_projections(self) -> pd.DataFrame:
        K    = self.biz["display_k"]
        P    = self.biz["avg_item_price_inr"]
        N    = self.biz["daily_sessions"]
        C2O  = self.biz["baseline_c2o_rate"]
        C2O_U= self.biz["c2o_uplift_per_hit"]

        ctrl_recall  = self.ctrl_grp[f"recall@{K}"].mean()
        treat_recall = self.treat_grp[f"recall@{K}"].mean()
        ctrl_prec    = self.ctrl_grp[f"precision@{K}"].mean()
        treat_prec   = self.treat_grp[f"precision@{K}"].mean()

        ctrl_attach  = ctrl_prec  * K
        treat_attach = treat_prec * K
        ctrl_aov     = ctrl_recall  * ctrl_attach  * P
        treat_aov    = treat_recall * treat_attach * P
        ctrl_c2o     = C2O + ctrl_recall  * C2O_U
        treat_c2o    = C2O + treat_recall * C2O_U
        daily_lift   = (treat_aov - ctrl_aov) * N

        rows = [
            ("Acceptance Rate (Recall@8)",
             f"{ctrl_recall:.4f}", f"{treat_recall:.4f}",
             f"{(treat_recall - ctrl_recall) / ctrl_recall * 100:+.1f}%"),
            ("Attach Rate (items/impression)",
             f"{ctrl_attach:.4f}", f"{treat_attach:.4f}",
             f"{(treat_attach - ctrl_attach) / ctrl_attach * 100:+.1f}%"),
            ("AOV Lift per Session (Rs.)",
             f"Rs.{ctrl_aov:.2f}", f"Rs.{treat_aov:.2f}",
             f"Rs.{treat_aov - ctrl_aov:+.2f}"),
            ("AOV lift as % of baseline",
             "—", "—",
             f"{(treat_aov - ctrl_aov) / self.biz['baseline_avg_order_inr'] * 100:+.2f}%"),
            ("C2O Rate Proxy",
             f"{ctrl_c2o:.4f}", f"{treat_c2o:.4f}",
             f"{(treat_c2o - ctrl_c2o) / ctrl_c2o * 100:+.2f}%"),
            ("Daily Revenue Uplift (Rs.)",
             "—", f"Rs.{daily_lift:,.0f}",
             f"at {N:,.0f} sessions/day"),
        ]
        return pd.DataFrame(rows, columns=["metric", "control", "treatment", "delta"])


# ── Online Experiment Design ─────────────────────────────────────────────────────

@dataclass
class ExperimentDesign:
    """
    Online A/B experiment design: sample size, MDE, duration.

    Parameters
    ----------
    ctrl_attach_rate : baseline attach rate (Recall@8)
    alpha            : significance level (default 0.05)
    power            : statistical power (default 0.80)
    traffic_split    : fraction of traffic per arm (default 0.50)
    daily_sessions   : daily CSAO impressions (default 5,000,000)
    min_days         : minimum experiment duration in days
    """
    ctrl_attach_rate: float = 0.6389
    alpha:            float = 0.05
    power:            float = 0.80
    traffic_split:    float = 0.50
    daily_sessions:   int   = 5_000_000
    min_days:         int   = 14
    max_days:         int   = 21

    def sample_size(self, mde: float) -> int:
        """
        Required sample size per arm for a given MDE (absolute).
        Uses two-proportion z-test formula.
        """
        p_t      = self.ctrl_attach_rate + mde
        p_pooled = (self.ctrl_attach_rate + p_t) / 2
        z_alpha  = norm.ppf(1 - self.alpha / 2)
        z_beta   = norm.ppf(self.power)
        n = (
            z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled))
            + z_beta * np.sqrt(
                self.ctrl_attach_rate * (1 - self.ctrl_attach_rate)
                + p_t * (1 - p_t)
            )
        ) ** 2 / mde ** 2
        return int(np.ceil(n))

    def detectable_mde(self, n_per_arm: int) -> float:
        """Binary-search for the smallest detectable MDE at a given sample size."""
        lo, hi = 0.0001, 0.5
        for _ in range(100):
            mid   = (lo + hi) / 2
            n_req = self.sample_size(mid)
            if n_req > n_per_arm:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def duration_days(self, n_per_arm: int) -> float:
        """Estimate experiment duration in days."""
        total_n = 2 * n_per_arm
        return total_n / (self.daily_sessions * self.traffic_split * 2)

    def sample_size_table(
        self,
        mde_scenarios: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Build sample size table for multiple MDE scenarios."""
        if mde_scenarios is None:
            mde_scenarios = [0.005, 0.010, 0.015, 0.020, 0.030]

        rows = []
        for mde in mde_scenarios:
            n     = self.sample_size(mde)
            total = 2 * n
            days  = self.duration_days(n)
            rows.append({
                "mde_abs":   mde,
                "mde_rel":   f"{mde / self.ctrl_attach_rate * 100:.1f}%",
                "n_per_arm": n,
                "total_n":   total,
                "days":      round(days, 1),
            })
        return pd.DataFrame(rows)

    def print_sample_size_table(
        self,
        mde_scenarios: Optional[List[float]] = None,
        observed_effect: Optional[float] = None,
    ) -> None:
        """Print the sample size / power analysis table."""
        print(f"\n{'='*70}")
        print("  ONLINE A/B EXPERIMENT DESIGN — Sample Size Analysis")
        print(f"{'='*70}")
        print(f"  Control attach rate : {self.ctrl_attach_rate:.4f}")
        print(f"  α={self.alpha}  Power={self.power}  Split={self.traffic_split:.0%}/{self.traffic_split:.0%}")
        print()
        print(f"  {'MDE (abs)':<12} {'MDE (rel)':<12} {'N per arm':>12} {'Total N':>12} {'Days':>8}")
        print("  " + "─" * 58)

        tbl = self.sample_size_table(mde_scenarios)
        for _, row in tbl.iterrows():
            flag = " ← rec." if self.min_days <= row["days"] <= self.max_days else ""
            print(
                f"  {row['mde_abs']:.3f}       {row['mde_rel']:>8}"
                f"  {row['n_per_arm']:>12,}  {row['total_n']:>12,}  {row['days']:>6.1f}d{flag}"
            )

        # 2-week analysis
        n_2w  = int(self.daily_sessions * self.traffic_split * self.min_days)
        mde_2w = self.detectable_mde(n_2w)
        print(f"\n  At {self.min_days}-day run ({n_2w/1e6:.1f}M users/arm):")
        print(f"    Detectable MDE = {mde_2w:.4f}  ({mde_2w/self.ctrl_attach_rate*100:.2f}% relative)")

        if observed_effect is not None:
            flag = "🟢 DETECTABLE" if observed_effect > mde_2w else "🔴 NOT DETECTABLE"
            print(f"    Observed lift  = {observed_effect:.4f}  → {flag}")

        print(f"{'='*70}\n")

    def print_framework(
        self,
        treat_attach_rate: Optional[float] = None,
    ) -> None:
        """Print the full online A/B test framework design document."""
        observed = (
            f"{treat_attach_rate - self.ctrl_attach_rate:+.4f}"
            if treat_attach_rate else "N/A"
        )
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║          CSAO RECOMMENDATION SYSTEM — ONLINE A/B TEST DESIGN        ║
╚══════════════════════════════════════════════════════════════════════╝

EXPERIMENT OVERVIEW
───────────────────
  Hypothesis   : LightGBM Binary v1 ranker will increase CSAO attach rate
                 vs retrieval_score baseline by ≥1.5pp absolute
  Control (A)  : Stage-1 retrieval_score ranking (no ML ranker)
  Treatment (B): LightGBM Binary v1 ranker on top of Stage-1 candidates

RANDOMISATION
─────────────
  Unit  : user_id (user-level, NOT session-level)
  Why   : Avoids same user seeing both arms → SUTVA violation
  Split : 50% Control / 50% Treatment
  Hash  : MD5(user_id + experiment_id) % 100 < 50 → Control

DURATION
────────
  Minimum  : 14 days (2 full weekly cycles to capture weekday+weekend)
  Maximum  : 21 days (novelty effect typically subsides after 3 weeks)
  Ramp     : Day 1 at 5% traffic for SLO check → Day 2 full 50% traffic

PRIMARY METRIC
──────────────
  CSAO Rail Attach Rate = sessions with ≥1 rail item added
                        / sessions where rail was displayed
  Success: statistically significant lift (α=0.05, power=0.80) AND ≥1pp absolute

SECONDARY METRICS
─────────────────
  1. AOV lift (Rs.)            — incremental order value per session
  2. Avg items per order       — depth of add-on acceptance
  3. CSAO order share          — % orders with ≥1 CSAO rail item
  4. Cart-to-Order (C2O) ratio — % cart sessions completing checkout
  5. Click-through rate        — % impressions with ≥1 rail click

GUARDRAIL METRICS (must NOT degrade)
─────────────────────────────────────
  ❗ Cart abandonment   : < +0.5pp increase
  ❗ Session duration   : < +10% increase
  ❗ Order completion   : < −0.2pp drop
  ❗ App error rate     : must not exceed baseline P99.9
  ❗ Inference latency  : P99 must stay < 300ms

ROLLBACK TRIGGERS
─────────────────
  Auto-rollback if ANY guardrail breached AND result is statistically
  significant (p < 0.01). Manual review for borderline cases.

STATISTICAL TESTING
───────────────────
  Primary    : Two-proportion z-test (attach rate is a proportion)
  Secondary  : Welch's t-test for continuous metrics (AOV)
  Peeking    : Sequential testing / Bonferroni-Holm to control false discovery rate
               — lock results until Day 14, no early stopping based on p-value
""")


# ── Guardrail Checker ────────────────────────────────────────────────────────────

@dataclass
class GuardrailConfig:
    """Thresholds for guardrail metrics."""
    max_cart_abandonment_increase_pp: float = 0.5
    max_session_duration_increase_pct: float = 10.0
    max_order_completion_drop_pp: float = 0.2
    max_inference_p99_ms: float = 300.0


def check_guardrails(
    control_metrics:   dict,
    treatment_metrics: dict,
    config: GuardrailConfig = GuardrailConfig(),
) -> pd.DataFrame:
    """
    Evaluate whether guardrail metrics are breached.

    Parameters
    ----------
    control_metrics   : dict of {metric_name: value} for control arm
    treatment_metrics : dict of {metric_name: value} for treatment arm

    Returns
    -------
    pd.DataFrame with columns: metric, control, treatment, delta, breached
    """
    checks = [
        (
            "Cart Abandonment Rate",
            control_metrics.get("cart_abandonment_rate", 0.0),
            treatment_metrics.get("cart_abandonment_rate", 0.0),
            lambda d: (d * 100) > config.max_cart_abandonment_increase_pp,
        ),
        (
            "Session Duration (s)",
            control_metrics.get("session_duration_s", 0.0),
            treatment_metrics.get("session_duration_s", 0.0),
            lambda d: (d / (control_metrics.get("session_duration_s", 1.0) + 1e-9) * 100)
                      > config.max_session_duration_increase_pct,
        ),
        (
            "Order Completion Rate",
            control_metrics.get("order_completion_rate", 0.0),
            treatment_metrics.get("order_completion_rate", 0.0),
            lambda d: (abs(d) * 100) > config.max_order_completion_drop_pp and d < 0,
        ),
        (
            "Inference P99 (ms)",
            control_metrics.get("inference_p99_ms", 0.0),
            treatment_metrics.get("inference_p99_ms", 0.0),
            lambda _: treatment_metrics.get("inference_p99_ms", 0.0)
                      > config.max_inference_p99_ms,
        ),
    ]

    rows = []
    all_clear = True
    for name, ctrl, treat, breach_fn in checks:
        delta    = treat - ctrl
        breached = breach_fn(delta)
        if breached:
            all_clear = False
        rows.append({
            "metric":   name,
            "control":  ctrl,
            "treatment":treat,
            "delta":    delta,
            "breached": breached,
        })

    df = pd.DataFrame(rows)
    if all_clear:
        logger.info("✅ All guardrails passed — experiment safe to run to conclusion.")
    else:
        n_breached = df["breached"].sum()
        logger.warning(
            "❗ %d guardrail(s) breached — consider rollback.", n_breached
        )
    return df
