"""
src/inference_pipeline.py
==========================
End-to-end online inference pipeline for the CSAO Rail.

Accepts a user cart event and returns the top-K ranked add-on
recommendations in <200ms wall time.

Pipeline steps
--------------
1. Validate and parse incoming cart event
2. Fetch real-time features (cart state, user history, item attributes)
3. Run Stage-1 candidate retrieval (co-occurrence + ctx + rule-fill)
4. Build feature matrix for candidates
5. Score candidates with LightGBM ranker
6. Return top-K ranked item IDs with scores

Usage
-----
    from src.inference_pipeline import InferencePipeline

    pipeline = InferencePipeline.from_artifacts(
        model_path   = "models/csao_ranker.lgb",
        cooc_path    = "data/candidate_generation/cooc_top.csv",
        ctx_path     = "data/candidate_generation/ctx_pop_top.csv",
        items_path   = "data/raw/items_clean.csv",
        users_path   = "data/raw/users.csv",
        sessions_path= "data/raw/sessions_clean.csv",
    )

    result = pipeline.predict(cart_event={
        "session_id":  12345,
        "user_id":     67890,
        "restaurant_id": 111,
        "step_number": 2,
        "cart_item_ids": [772, 815],
        "meal_time_bucket": "lunch",
        "is_weekend": 0,
        "city": "Mumbai",
    }, top_k=8)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ITEM_CATEGORIES = ["main", "beverage", "dessert", "side"]
RULE_TRIGGER_CATS = ["beverage", "dessert", "side"]

# ── Data structures ──────────────────────────────────────────────────────────────

@dataclass
class CartEvent:
    """Parsed incoming cart event from the CSAO rail trigger."""
    session_id:       int
    user_id:          int
    restaurant_id:    int
    step_number:      int
    cart_item_ids:    List[int]
    meal_time_bucket: str
    is_weekend:       int       = 0
    city:             str       = ""
    cart_total_value: float     = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "CartEvent":
        return cls(
            session_id       = int(d["session_id"]),
            user_id          = int(d.get("user_id", 0)),
            restaurant_id    = int(d["restaurant_id"]),
            step_number      = int(d["step_number"]),
            cart_item_ids    = [int(x) for x in d.get("cart_item_ids", [])],
            meal_time_bucket = str(d.get("meal_time_bucket", "lunch")),
            is_weekend       = int(d.get("is_weekend", 0)),
            city             = str(d.get("city", "")),
            cart_total_value = float(d.get("cart_total_value", 0.0)),
        )


@dataclass
class Recommendation:
    """Single recommended item with its score and metadata."""
    item_id:    int
    score:      float
    rank:       int
    src_cooc:   int = 0
    src_ctx:    int = 0
    src_rule:   int = 0
    category:   str = ""
    price:      float = 0.0


@dataclass
class PredictionResult:
    """Output of one inference call."""
    session_id:      int
    step_number:     int
    recommendations: List[Recommendation]
    latency_ms:      float
    n_candidates:    int
    retrieval_sources: Dict[str, float] = field(default_factory=dict)


# ── Feature store abstraction ─────────────────────────────────────────────────────

class FeatureStore:
    """
    In-memory feature store.

    In production this would be backed by Redis (real-time cart state)
    and BigQuery / Hive (batch user/item features). For offline use
    and testing, it loads everything from CSV files at startup.
    """

    def __init__(
        self,
        items:    pd.DataFrame,
        users:    pd.DataFrame,
        sessions: Optional[pd.DataFrame] = None,
    ):
        # Item features: item_id → dict
        self._items = (
            items
            .set_index("item_id")
            .to_dict(orient="index")
        )

        # User features: user_id → dict
        self._users = (
            users
            .set_index("user_id")
            .to_dict(orient="index")
        ) if users is not None else {}

        # Item category lookup
        self._item_cat: Dict[int, str] = {
            int(k): str(v.get("effective_category", "unknown"))
            for k, v in self._items.items()
        }
        self._item_price: Dict[int, float] = {
            int(k): float(v.get("price", 0.0))
            for k, v in self._items.items()
        }

    def get_item_features(self, item_id: int) -> dict:
        return dict(self._items.get(item_id, {}))

    def get_user_features(self, user_id: int) -> dict:
        return dict(self._users.get(user_id, {}))

    def item_category(self, item_id: int) -> str:
        return self._item_cat.get(item_id, "unknown")

    def item_price(self, item_id: int) -> float:
        return self._item_price.get(item_id, 0.0)


# ── Retrieval index ───────────────────────────────────────────────────────────────

class RetrievalIndex:
    """
    Serves pre-built co-occurrence and context-popularity lookups.
    Loaded once at startup; queries are O(1) dict lookups.
    """

    def __init__(
        self,
        cooc_df:        pd.DataFrame,
        ctx_df:         pd.DataFrame,
        items_by_rest:  Dict[int, np.ndarray],
        items_by_rest_cat: Dict[tuple, np.ndarray],
        cooc_weight:    float = 0.7,
        ctx_weight:     float = 0.3,
        rule_boost:     float = 0.05,
        n_candidates:   int   = 30,
        rule_top:       int   = 50,
    ):
        self.cooc_weight   = cooc_weight
        self.ctx_weight    = ctx_weight
        self.rule_boost    = rule_boost
        self.n_candidates  = n_candidates
        self.rule_top      = rule_top

        self._items_by_rest     = items_by_rest
        self._items_by_rest_cat = items_by_rest_cat

        # Build in-memory lookups
        self._cooc: Dict[int, Dict[int, float]] = defaultdict(dict)
        for row in cooc_df.itertuples(index=False):
            self._cooc[int(row.anchor_item_id)][int(row.candidate_item_id)] = float(row.cooc_score)

        self._ctx: Dict[tuple, Dict[int, float]] = defaultdict(dict)
        for row in ctx_df.itertuples(index=False):
            self._ctx[(int(row.restaurant_id), str(row.meal_time_bucket))][
                int(row.candidate_item_id)
            ] = float(row.ctx_score)

        logger.info(
            "RetrievalIndex loaded: %d cooc anchors, %d ctx buckets",
            len(self._cooc), len(self._ctx),
        )

    def retrieve(
        self,
        cart_item_ids:  List[int],
        restaurant_id:  int,
        meal_time:      str,
        cart_cats:      set,
        item_cat_fn,            # callable: item_id → category string
    ) -> List[Dict[str, Any]]:
        """
        Return top-N candidates with retrieval scores and source flags.

        Parameters
        ----------
        cart_item_ids : items currently in cart
        restaurant_id : current restaurant
        meal_time     : meal time bucket string
        cart_cats     : set of categories already in cart
        item_cat_fn   : function to look up item category

        Returns
        -------
        list of dicts: {candidate_item_id, retrieval_score, src_cooc, src_ctx, src_rule}
        """
        cart_set  = set(cart_item_ids)
        rest_pool = set(self._items_by_rest.get(restaurant_id, np.array([], dtype=int)).tolist())

        cooc_scores: Dict[int, float] = {}
        for anchor_id in cart_item_ids:
            for cand_id, s in self._cooc.get(anchor_id, {}).items():
                if cand_id not in cart_set and cand_id in rest_pool:
                    cooc_scores[cand_id] = max(
                        cooc_scores.get(cand_id, 0.0),
                        s * self.cooc_weight,
                    )

        ctx_scores: Dict[int, float] = {
            cid: s
            for cid, s in self._ctx.get((restaurant_id, meal_time), {}).items()
            if cid not in cart_set and cid in rest_pool
        }

        rule_scores: Dict[int, float] = {}
        for trigger_cat in RULE_TRIGGER_CATS:
            if trigger_cat not in cart_cats:
                pool = self._items_by_rest_cat.get(
                    (restaurant_id, trigger_cat), np.array([], dtype=int)
                )
                for cid in pool[: self.rule_top]:
                    if cid not in cart_set and cid in rest_pool:
                        rule_scores[cid] = rule_scores.get(cid, 0.0) + self.rule_boost

        all_cands: Dict[int, float] = {}
        for cid in set(list(cooc_scores) + list(ctx_scores) + list(rule_scores)):
            all_cands[cid] = (
                cooc_scores.get(cid, 0.0)
                + ctx_scores.get(cid, 0.0) * self.ctx_weight
                + rule_scores.get(cid, 0.0)
            )

        ranked = sorted(all_cands.items(), key=lambda x: -x[1])
        top_n  = ranked[: self.n_candidates]

        return [
            {
                "candidate_item_id": int(cid),
                "retrieval_score":   float(score),
                "src_cooc":          int(cid in cooc_scores),
                "src_ctx":           int(cid in ctx_scores),
                "src_rule":          int(cid in rule_scores),
            }
            for cid, score in top_n
        ]


# ── Feature assembler ─────────────────────────────────────────────────────────────

class FeatureAssembler:
    """Builds the 47-feature vector for each candidate given a CartEvent."""

    # Must match EXACTLY the model's feature_name() order
    FEATURE_COLS = [
        "retrieval_score", "src_cooc", "src_ctx", "src_rule",
        "is_weekend", "cart_item_count", "cart_total_value",
        "cart_has_main", "cart_has_beverage", "cart_has_dessert", "cart_has_side",
        "missing_beverage_flag", "missing_dessert_flag", "missing_side_flag",
        "price", "popularity_score", "historical_attach_rate",
        "order_count_90d", "recency_days", "avg_order_value",
        "veg_preference_ratio", "dessert_affinity_score", "beverage_affinity_score",
        "price_sensitivity_score", "aggregate_rating", "overall_attach_rate",
        "meal_time_specificity", "meal_time_overlap",
        "step_decay",
        "x_missing_bev_aff", "x_missing_des_aff",
        "x_price_sens_price", "x_cart_value_price",
        "retrieval_x_attach", "retrieval_x_popularity",
        "missing_bev_x_retrieval", "missing_des_x_retrieval",
        "last_item_category_beverage", "last_item_category_dessert",
        "last_item_category_main", "last_item_category_side",
        "effective_category_beverage", "effective_category_dessert",
        "effective_category_main", "effective_category_side",
        "x_cand_bev_aff", "x_cand_des_aff",
    ]

    def __init__(self, feature_store: FeatureStore):
        self.fs = feature_store

    def build(
        self,
        event:      CartEvent,
        candidates: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Build the feature matrix for all candidates of one cart event.

        Returns
        -------
        pd.DataFrame with one row per candidate, columns = FEATURE_COLS
        """
        # ── Cart-state features (constant across candidates) ──────────────────
        cart_cats = {self.fs.item_category(iid) for iid in event.cart_item_ids}
        cart_has  = {cat: int(cat in cart_cats) for cat in ITEM_CATEGORIES}
        last_cat  = (
            self.fs.item_category(event.cart_item_ids[-1])
            if event.cart_item_ids else "unknown"
        )

        cart_total = (
            event.cart_total_value
            if event.cart_total_value > 0
            else sum(self.fs.item_price(iid) for iid in event.cart_item_ids)
        )

        # ── User features ─────────────────────────────────────────────────────
        uf = self.fs.get_user_features(event.user_id)
        bev_aff  = float(uf.get("beverage_affinity_score", 0.0))
        des_aff  = float(uf.get("dessert_affinity_score",  0.0))
        price_s  = float(uf.get("price_sensitivity_score", 0.0))
        o90d     = float(uf.get("order_count_90d",         0.0))
        recency  = float(uf.get("recency_days",            0.0))
        aov      = float(uf.get("avg_order_value",         0.0))
        veg_r    = float(uf.get("veg_preference_ratio",    0.0))

        step_decay     = 1.0 / max(event.step_number, 1)
        miss_bev       = 1 - cart_has["beverage"]
        miss_des       = 1 - cart_has["dessert"]
        miss_side      = 1 - cart_has["side"]

        rows = []
        for cand in candidates:
            cid  = cand["candidate_item_id"]
            ret  = cand["retrieval_score"]
            itf  = self.fs.get_item_features(cid)

            price        = float(itf.get("price",                    0.0))
            pop          = float(itf.get("popularity_score",         0.0))
            attach       = float(itf.get("historical_attach_rate",   0.0))
            rating       = float(itf.get("aggregate_rating",         0.0))
            o_attach     = float(itf.get("overall_attach_rate",      0.0))
            mt_spec      = float(itf.get("meal_time_specificity",    0.0))
            mt_overlap   = float(itf.get("meal_time_overlap",        0.0))
            eff_cat      = self.fs.item_category(cid)

            # Category one-hot
            eff_bev  = int(eff_cat == "beverage")
            eff_des  = int(eff_cat == "dessert")
            eff_main = int(eff_cat == "main")
            eff_side = int(eff_cat == "side")

            # Last-item category one-hot
            lc_bev  = int(last_cat == "beverage")
            lc_des  = int(last_cat == "dessert")
            lc_main = int(last_cat == "main")
            lc_side = int(last_cat == "side")

            rows.append({
                "retrieval_score":            ret,
                "src_cooc":                   cand["src_cooc"],
                "src_ctx":                    cand["src_ctx"],
                "src_rule":                   cand["src_rule"],
                "is_weekend":                 event.is_weekend,
                "cart_item_count":            len(event.cart_item_ids),
                "cart_total_value":           cart_total,
                "cart_has_main":              cart_has["main"],
                "cart_has_beverage":          cart_has["beverage"],
                "cart_has_dessert":           cart_has["dessert"],
                "cart_has_side":              cart_has["side"],
                "missing_beverage_flag":      miss_bev,
                "missing_dessert_flag":       miss_des,
                "missing_side_flag":          miss_side,
                "price":                      price,
                "popularity_score":           pop,
                "historical_attach_rate":     attach,
                "order_count_90d":            o90d,
                "recency_days":               recency,
                "avg_order_value":            aov,
                "veg_preference_ratio":       veg_r,
                "dessert_affinity_score":     des_aff,
                "beverage_affinity_score":    bev_aff,
                "price_sensitivity_score":    price_s,
                "aggregate_rating":           rating,
                "overall_attach_rate":        o_attach,
                "meal_time_specificity":      mt_spec,
                "meal_time_overlap":          mt_overlap,
                "step_decay":                 step_decay,
                "x_missing_bev_aff":          miss_bev * bev_aff,
                "x_missing_des_aff":          miss_des * des_aff,
                "x_price_sens_price":         price_s  * price,
                "x_cart_value_price":         cart_total * price,
                "retrieval_x_attach":         ret * attach,
                "retrieval_x_popularity":     ret * pop,
                "missing_bev_x_retrieval":    miss_bev * ret,
                "missing_des_x_retrieval":    miss_des * ret,
                "last_item_category_beverage":lc_bev,
                "last_item_category_dessert": lc_des,
                "last_item_category_main":    lc_main,
                "last_item_category_side":    lc_side,
                "effective_category_beverage":eff_bev,
                "effective_category_dessert": eff_des,
                "effective_category_main":    eff_main,
                "effective_category_side":    eff_side,
                "x_cand_bev_aff":             eff_bev * bev_aff,
                "x_cand_des_aff":             eff_des * des_aff,
            })

        df = pd.DataFrame(rows, columns=self.FEATURE_COLS)
        return df.fillna(0.0)


# ── Main pipeline ─────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    End-to-end CSAO rail inference pipeline.

    Parameters
    ----------
    ranker    : loaded CSAORanker (or any object with .predict(df) → array)
    retrieval : RetrievalIndex instance
    assembler : FeatureAssembler instance
    top_k     : default number of recommendations to return
    """

    def __init__(
        self,
        ranker:    Any,
        retrieval: RetrievalIndex,
        assembler: FeatureAssembler,
        top_k:     int = 8,
    ):
        self.ranker    = ranker
        self.retrieval = retrieval
        self.assembler = assembler
        self.top_k     = top_k

    # ── factory constructor ─────────────────────────────────────────────────────

    @classmethod
    def from_artifacts(
        cls,
        model_path:    str = "models/csao_ranker.lgb",
        cooc_path:     str = "data/candidate_generation/cooc_top.csv",
        ctx_path:      str = "data/candidate_generation/ctx_pop_top.csv",
        items_path:    str = "data/raw/items_clean.csv",
        users_path:    str = "data/raw/users.csv",
        sessions_path: Optional[str] = None,
        top_k:         int = 8,
        retrieval_config: Optional[dict] = None,
    ) -> "InferencePipeline":
        """
        Load all artifacts from disk and build a ready-to-serve pipeline.
        """
        from src.ranking_model import CSAORanker

        logger.info("Loading inference pipeline artifacts ...")
        t0 = time.perf_counter()

        # Load data
        items    = cls._load_items(items_path)
        users    = pd.read_csv(users_path) if users_path and Path(users_path).exists() else pd.DataFrame()
        cooc_df  = pd.read_csv(cooc_path)
        ctx_df   = pd.read_csv(ctx_path)

        # Build item index for retrieval
        items_by_rest, items_by_rest_cat = cls._build_item_index(items)

        # Build components
        cfg       = retrieval_config or {}
        retrieval = RetrievalIndex(
            cooc_df            = cooc_df,
            ctx_df             = ctx_df,
            items_by_rest      = items_by_rest,
            items_by_rest_cat  = items_by_rest_cat,
            cooc_weight        = cfg.get("cooc_weight",   0.7),
            ctx_weight         = cfg.get("ctx_weight",    0.3),
            rule_boost         = cfg.get("rule_boost",    0.05),
            n_candidates       = cfg.get("n_candidates",  30),
            rule_top           = cfg.get("rule_top",      50),
        )
        fs        = FeatureStore(items=items, users=users)
        assembler = FeatureAssembler(feature_store=fs)
        ranker    = CSAORanker.load(model_path)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Pipeline ready in %.1fms", elapsed)
        return cls(ranker=ranker, retrieval=retrieval, assembler=assembler, top_k=top_k)

    # ── public predict ──────────────────────────────────────────────────────────

    def predict(
        self,
        cart_event: dict | CartEvent,
        top_k:      Optional[int] = None,
    ) -> PredictionResult:
        """
        Score candidates for a single cart event and return top-K recommendations.

        Parameters
        ----------
        cart_event : dict or CartEvent describing the current cart state
        top_k      : override default top-K

        Returns
        -------
        PredictionResult with ranked recommendations and latency info
        """
        t_start = time.perf_counter()

        if isinstance(cart_event, dict):
            event = CartEvent.from_dict(cart_event)
        else:
            event = cart_event

        k = top_k or self.top_k

        # ── 1. Retrieve candidates ────────────────────────────────────────────
        cart_cats = {
            self.assembler.fs.item_category(iid)
            for iid in event.cart_item_ids
        }
        candidates = self.retrieval.retrieve(
            cart_item_ids = event.cart_item_ids,
            restaurant_id = event.restaurant_id,
            meal_time     = event.meal_time_bucket,
            cart_cats     = cart_cats,
            item_cat_fn   = self.assembler.fs.item_category,
        )

        if not candidates:
            logger.warning(
                "No candidates retrieved for session=%d step=%d",
                event.session_id, event.step_number,
            )
            return PredictionResult(
                session_id=event.session_id,
                step_number=event.step_number,
                recommendations=[],
                latency_ms=(time.perf_counter() - t_start) * 1000,
                n_candidates=0,
            )

        # ── 2. Assemble features ──────────────────────────────────────────────
        feat_df = self.assembler.build(event, candidates)

        # ── 3. Score ──────────────────────────────────────────────────────────
        scores = self.ranker.model.predict(
            feat_df.values.astype(np.float32),
            num_iteration=self.ranker.model.best_iteration,
        )

        # ── 4. Rank and return top-K ──────────────────────────────────────────
        ranked_idx = np.argsort(scores)[::-1]
        top_idx    = ranked_idx[:k]

        recommendations = []
        for rank, idx in enumerate(top_idx, start=1):
            cand = candidates[idx]
            cid  = cand["candidate_item_id"]
            recommendations.append(Recommendation(
                item_id  = cid,
                score    = float(scores[idx]),
                rank     = rank,
                src_cooc = cand["src_cooc"],
                src_ctx  = cand["src_ctx"],
                src_rule = cand["src_rule"],
                category = self.assembler.fs.item_category(cid),
                price    = self.assembler.fs.item_price(cid),
            ))

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Coverage by source
        n = len(candidates)
        src_cov = {
            "cooc": sum(c["src_cooc"] for c in candidates) / n,
            "ctx":  sum(c["src_ctx"]  for c in candidates) / n,
            "rule": sum(c["src_rule"] for c in candidates) / n,
        }

        logger.debug(
            "session=%d step=%d | %d candidates → top-%d | %.2fms",
            event.session_id, event.step_number, n, k, latency_ms,
        )

        return PredictionResult(
            session_id        = event.session_id,
            step_number       = event.step_number,
            recommendations   = recommendations,
            latency_ms        = latency_ms,
            n_candidates      = n,
            retrieval_sources = src_cov,
        )

    def predict_batch(
        self,
        cart_events: List[dict | CartEvent],
        top_k:       Optional[int] = None,
    ) -> List[PredictionResult]:
        """Score a list of cart events. Returns results in input order."""
        return [self.predict(e, top_k) for e in cart_events]

    def to_dict(self, result: PredictionResult) -> dict:
        """Serialise a PredictionResult to a JSON-compatible dict."""
        return {
            "session_id":   result.session_id,
            "step_number":  result.step_number,
            "latency_ms":   round(result.latency_ms, 3),
            "n_candidates": result.n_candidates,
            "retrieval_sources": result.retrieval_sources,
            "recommendations": [
                {
                    "rank":     r.rank,
                    "item_id":  r.item_id,
                    "score":    round(r.score, 4),
                    "category": r.category,
                    "price":    r.price,
                    "sources":  {
                        "cooc": bool(r.src_cooc),
                        "ctx":  bool(r.src_ctx),
                        "rule": bool(r.src_rule),
                    },
                }
                for r in result.recommendations
            ],
        }

    # ── private helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_items(path: str) -> pd.DataFrame:
        items = pd.read_csv(path)
        if "effective_category" not in items.columns and "normalized_category" in items.columns:
            items = items.rename(columns={"normalized_category": "effective_category"})
        items["item_id"] = items["item_id"].astype(int)
        return items

    @staticmethod
    def _build_item_index(
        items: pd.DataFrame,
    ) -> tuple[Dict[int, np.ndarray], Dict[tuple, np.ndarray]]:
        items = items.copy()
        items["pop_rank_score"] = (
            items.get("popularity_score", pd.Series(0.0, index=items.index)).fillna(0)
            + items.get("historical_attach_rate", pd.Series(0.0, index=items.index)).fillna(0)
        )
        items_sorted = items.sort_values(
            ["restaurant_id", "pop_rank_score"], ascending=[True, False]
        )
        by_rest = {
            int(rid): grp["item_id"].values
            for rid, grp in items_sorted.groupby("restaurant_id")
        }
        by_rest_cat = {
            (int(rid), cat): grp["item_id"].values
            for (rid, cat), grp in items_sorted.groupby(
                ["restaurant_id", "effective_category"]
            )
        }
        return by_rest, by_rest_cat
