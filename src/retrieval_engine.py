"""
src/retrieval_engine.py
========================
Stage 1: Candidate generation for the CSAO rail.

Three retrieval sources are fused into a single ranked candidate list
per (session_id, step_number) query group:

  1. Co-occurrence retrieval  — weighted item-item pairs from training carts
  2. Context popularity       — top items per (restaurant, meal_time_bucket)
  3. Rule-based fill          — category heuristics for cold-start / gap-fill

Output columns per candidate
-----------------------------
  session_id, step_number, restaurant_id, meal_time_bucket,
  candidate_item_id, retrieval_score, src_cooc, src_ctx, src_rule,
  label_addon_added

Mirrors notebooks/02_candidate_generation.ipynb

Usage
-----
    from src.retrieval_engine import RetrievalEngine
    engine = RetrievalEngine(items, cart, sessions)
    engine.fit(train_session_ids)               # build cooc + ctx tables
    candidates = engine.generate(session_ids)   # generate candidates
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Default hyperparameters (match notebook values) ────────────────────────────
DEFAULT_CONFIG = {
    "cooc_top_per_anchor": 100,   # top co-occurring items to keep per anchor
    "ctx_top_per_bucket":  50,    # top items per (restaurant, meal_time) bucket
    "n_candidates":        30,    # final candidate pool size per (session, step)
    "rule_top":            50,    # rule-fill pool size per category
    "cooc_weight":         0.7,   # blend weight for co-occurrence score
    "ctx_weight":          0.3,   # blend weight for context-popularity score
    "rule_boost":          0.05,  # score boost added to rule-fill candidates
}

RULE_TRIGGER_CATS = ["beverage", "dessert", "side"]


class RetrievalEngine:
    """
    Two-stage CSAO candidate retrieval engine.

    Parameters
    ----------
    items    : items_clean DataFrame (must have item_id, restaurant_id,
               effective_category, price, popularity_score, historical_attach_rate)
    cart     : cart_events_clean DataFrame
    sessions : sessions_clean DataFrame (must have session_id, restaurant_id,
               meal_time_bucket)
    config   : dict of hyperparameters (see DEFAULT_CONFIG)
    """

    def __init__(
        self,
        items:    pd.DataFrame,
        cart:     pd.DataFrame,
        sessions: pd.DataFrame,
        config:   Optional[dict] = None,
    ):
        self.cfg      = {**DEFAULT_CONFIG, **(config or {})}
        self.items    = items.copy()
        self.cart     = cart.copy()
        self.sessions = sessions.copy()

        # Will be populated by fit()
        self._cooc_lookup: dict = {}
        self._ctx_lookup:  dict = {}
        self._items_by_restaurant:  dict = {}
        self._items_by_rest_cat:    dict = {}
        self._sess_map: Optional[pd.DataFrame] = None
        self._fitted = False

        # Pre-compute item index structures
        self._build_item_index()

    # ── public ─────────────────────────────────────────────────────────────────

    def fit(self, train_session_ids: Iterable[int]) -> "RetrievalEngine":
        """
        Build co-occurrence and context-popularity tables from training sessions.

        Parameters
        ----------
        train_session_ids : iterable of session_id values used for training

        Returns
        -------
        self
        """
        train_ids = set(train_session_ids)
        logger.info("Fitting retrieval engine on %d training sessions ...", len(train_ids))

        self._sess_map = (
            self.sessions
            .set_index("session_id")[["restaurant_id", "meal_time_bucket"]]
        )

        cart_train = (
            self.cart[self.cart["session_id"].isin(train_ids)]
            .copy()
            .merge(
                self._sess_map[["restaurant_id"]],
                left_on="session_id", right_index=True, how="left",
            )
        )

        self._build_cooc_table(cart_train, train_ids)
        self._build_ctx_table(cart_train)
        self._fitted = True
        logger.info("RetrievalEngine fitted.")
        return self

    def generate(
        self,
        session_ids: Iterable[int],
        cart_df:     Optional[pd.DataFrame] = None,
        labels_df:   Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate candidates for a set of sessions.

        Parameters
        ----------
        session_ids : sessions to generate candidates for
        cart_df     : cart events for these sessions (defaults to self.cart)
        labels_df   : optional DataFrame with (session_id, step_number, item_id)
                      used as positive labels (label_addon_added = 1)

        Returns
        -------
        pd.DataFrame with columns:
            session_id, step_number, restaurant_id, meal_time_bucket,
            candidate_item_id, retrieval_score, src_cooc, src_ctx, src_rule,
            label_addon_added
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before generate()")

        session_ids = set(session_ids)
        cart = (cart_df if cart_df is not None else self.cart).copy()
        cart = cart[cart["session_id"].isin(session_ids)].sort_values(
            ["session_id", "step_number"]
        )

        # Build per-session cart lookup
        cart_by_session: dict = {}
        for sid, grp in cart.groupby("session_id", sort=False):
            g = grp.sort_values("step_number")
            cart_by_session[sid] = (g["step_number"].values, g["item_id"].values)

        candidate_rows = []

        for sid in tqdm(sorted(session_ids), desc="Generating candidates"):
            if sid not in cart_by_session:
                continue

            steps, iids = cart_by_session[sid]
            n = len(iids)
            if n < 2:
                continue  # need ≥2 items: at least 1 in cart, 1 as label target

            try:
                rest_id   = int(self._sess_map.at[sid, "restaurant_id"])
                meal_time = self._sess_map.at[sid, "meal_time_bucket"]
            except KeyError:
                continue

            rest_pool = self._items_by_restaurant.get(rest_id, np.array([], dtype=int))

            # FIX: loop range(0, n-1) — last step has no future label
            for step_idx in range(n - 1):
                current_step  = int(steps[step_idx])
                next_item_id  = int(iids[step_idx + 1])
                cart_so_far   = set(iids[:step_idx + 1].tolist())

                # ── Source 1: Co-occurrence ───────────────────────────────────
                cooc_scores: dict = {}
                for anchor_id in cart_so_far:
                    for cand_id, score in self._cooc_lookup.get(anchor_id, {}).items():
                        if cand_id not in cart_so_far:
                            cooc_scores[cand_id] = max(
                                cooc_scores.get(cand_id, 0.0),
                                score * self.cfg["cooc_weight"],
                            )

                # ── Source 2: Context popularity ──────────────────────────────
                ctx_scores: dict = self._ctx_lookup.get((rest_id, meal_time), {})

                # ── Source 3: Rule-based fill ─────────────────────────────────
                rule_scores: dict = {}
                cart_cats = {
                    self.items.set_index("item_id")
                    .get("effective_category", pd.Series(dtype="object"))
                    .get(iid, "unknown")
                    for iid in cart_so_far
                }
                for trigger_cat in RULE_TRIGGER_CATS:
                    if trigger_cat not in cart_cats:
                        pool = self._items_by_rest_cat.get(
                            (rest_id, trigger_cat), np.array([], dtype=int)
                        )
                        for cand_id in pool[: self.cfg["rule_top"]]:
                            if cand_id not in cart_so_far:
                                rule_scores[cand_id] = (
                                    rule_scores.get(cand_id, 0.0) + self.cfg["rule_boost"]
                                )

                # ── Merge and rank ─────────────────────────────────────────────
                all_candidates: dict = {}
                for cand_id in set(
                    list(cooc_scores) + list(ctx_scores) + list(rule_scores)
                ):
                    if cand_id in cart_so_far:
                        continue
                    if int(cand_id) not in set(rest_pool.tolist()):
                        continue  # must belong to the same restaurant
                    score = (
                        cooc_scores.get(cand_id, 0.0)
                        + ctx_scores.get(cand_id, 0.0) * self.cfg["ctx_weight"]
                        + rule_scores.get(cand_id, 0.0)
                    )
                    all_candidates[cand_id] = score

                # Guarantee the next item is always in the candidate pool (Fix 5)
                if next_item_id not in all_candidates:
                    all_candidates[next_item_id] = self.cfg["rule_boost"]

                # Sort by score descending and take top N
                ranked = sorted(all_candidates.items(), key=lambda x: -x[1])
                top_n  = ranked[: self.cfg["n_candidates"]]

                for cand_id, ret_score in top_n:
                    candidate_rows.append({
                        "session_id":        sid,
                        "step_number":       current_step,
                        "restaurant_id":     rest_id,
                        "meal_time_bucket":  meal_time,
                        "candidate_item_id": cand_id,
                        "retrieval_score":   float(ret_score),
                        "src_cooc":          int(cand_id in cooc_scores),
                        "src_ctx":           int(cand_id in ctx_scores),
                        "src_rule":          int(cand_id in rule_scores),
                        "label_addon_added": int(cand_id == next_item_id),
                    })

        result = pd.DataFrame(candidate_rows)
        logger.info(
            "Generated %d candidate rows across %d sessions",
            len(result),
            result["session_id"].nunique() if not result.empty else 0,
        )
        return result

    def save_tables(self, cooc_path: str = "cooc_top.csv",
                    ctx_path: str = "ctx_pop_top.csv") -> None:
        """Persist the co-occurrence and context tables to CSV."""
        self._cooc_df.to_csv(cooc_path, index=False)
        self._ctx_df.to_csv(ctx_path, index=False)
        logger.info("Saved cooc → %s  ctx → %s", cooc_path, ctx_path)

    # ── private ─────────────────────────────────────────────────────────────────

    def _build_item_index(self) -> None:
        """Pre-build restaurant-level item pools sorted by pop_rank_score."""
        items = self.items.copy()
        items["pop_rank_score"] = (
            items["popularity_score"].fillna(0)
            + items["historical_attach_rate"].fillna(0)
        )
        items_sorted = items.sort_values(
            ["restaurant_id", "pop_rank_score"], ascending=[True, False]
        )

        self._items_by_restaurant = {
            rest_id: grp["item_id"].values
            for rest_id, grp in items_sorted.groupby("restaurant_id")
        }
        self._items_by_rest_cat = {
            (rest_id, cat): grp["item_id"].values
            for (rest_id, cat), grp in items_sorted.groupby(
                ["restaurant_id", "effective_category"]
            )
        }
        logger.debug(
            "Item index built: %d restaurants, %d (rest, cat) buckets",
            len(self._items_by_restaurant),
            len(self._items_by_rest_cat),
        )

    def _build_cooc_table(self, cart_train: pd.DataFrame, train_ids: Set[int]) -> None:
        """Build weighted co-occurrence table from training cart events."""
        cooc_accum: dict = defaultdict(float)

        for sid, grp in tqdm(
            cart_train.groupby("session_id", sort=False),
            desc="Building co-occurrence table",
            total=len(train_ids),
        ):
            rest_id = int(grp["restaurant_id"].iloc[0])
            steps   = grp["step_number"].values
            iids    = grp["item_id"].values
            n       = len(iids)
            if n < 2:
                continue
            for ti in range(n):
                for tj in range(ti + 1, n):
                    w = 1.0 / (1.0 + float(steps[tj] - steps[ti]))
                    cooc_accum[(rest_id, int(iids[ti]), int(iids[tj]))] += w

        cooc_df = pd.DataFrame(
            [(r, a, c, s) for (r, a, c), s in cooc_accum.items()],
            columns=["restaurant_id", "anchor_item_id", "candidate_item_id", "cooc_score"],
        )
        cooc_df = cooc_df.sort_values(
            ["restaurant_id", "anchor_item_id", "cooc_score"],
            ascending=[True, True, False],
        )
        cooc_df["cooc_rank"] = (
            cooc_df.groupby(["restaurant_id", "anchor_item_id"]).cumcount() + 1
        )
        cooc_top = cooc_df[cooc_df["cooc_rank"] <= self.cfg["cooc_top_per_anchor"]].copy()
        self._cooc_df = cooc_top

        # Build lookup dict: anchor_item_id → {candidate_item_id: cooc_score}
        self._cooc_lookup = defaultdict(dict)
        for row in cooc_top.itertuples(index=False):
            self._cooc_lookup[int(row.anchor_item_id)][int(row.candidate_item_id)] = (
                row.cooc_score
            )

        logger.info(
            "Co-occurrence table: %d rows, %d unique anchors",
            len(cooc_top),
            cooc_top["anchor_item_id"].nunique(),
        )

    def _build_ctx_table(self, cart_train: pd.DataFrame) -> None:
        """Build context-popularity table from training cart events."""
        cart_ctx = cart_train.merge(
            self._sess_map[["meal_time_bucket"]],
            left_on="session_id", right_index=True, how="left",
        )

        ctx_counts = (
            cart_ctx
            .groupby(["restaurant_id", "meal_time_bucket", "item_id"])
            .size()
            .reset_index(name="count")
        )
        ctx_counts["ctx_score"] = (
            ctx_counts["count"]
            / ctx_counts.groupby(["restaurant_id", "meal_time_bucket"])["count"]
            .transform("sum")
        )
        ctx_counts = ctx_counts.sort_values(
            ["restaurant_id", "meal_time_bucket", "ctx_score"],
            ascending=[True, True, False],
        )
        ctx_counts["ctx_rank"] = (
            ctx_counts.groupby(["restaurant_id", "meal_time_bucket"]).cumcount() + 1
        )
        ctx_top = (
            ctx_counts[ctx_counts["ctx_rank"] <= self.cfg["ctx_top_per_bucket"]]
            .rename(columns={"item_id": "candidate_item_id"})
            [["restaurant_id", "meal_time_bucket", "candidate_item_id", "ctx_score"]]
            .copy()
        )
        self._ctx_df = ctx_top

        # Build lookup dict: (restaurant_id, meal_time_bucket) → {cand_id: ctx_score}
        self._ctx_lookup = defaultdict(dict)
        for row in ctx_top.itertuples(index=False):
            self._ctx_lookup[(int(row.restaurant_id), row.meal_time_bucket)][
                int(row.candidate_item_id)
            ] = row.ctx_score

        logger.info(
            "Context-popularity table: %d rows, %d unique (rest, meal_time) buckets",
            len(ctx_top),
            ctx_top.groupby(["restaurant_id", "meal_time_bucket"]).ngroups,
        )
