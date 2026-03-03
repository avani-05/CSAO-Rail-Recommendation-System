"""
src/feature_pipeline.py
========================
Build the full 47-feature matrix from raw data tables.

Mirrors the logic in notebooks/01_split_and_feature_engineering.ipynb.

Pipeline stages
---------------
1. Cart-state features  (real-time at each step)
2. Session features
3. Item features        (candidate item attributes)
4. User features        (90-day behavioural history)
5. Restaurant features
6. Interaction features (cross-table multiplicative signals)
7. Column alignment     (ensure train/test have identical schema)

Usage
-----
    from src.feature_pipeline import FeaturePipeline
    pipeline = FeaturePipeline(items, users, sessions, cart, restaurants)
    train_feat = pipeline.build(rank_train, split="train")
    test_feat  = pipeline.build(rank_test,  split="test", reference=train_feat)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Columns to exclude from the model feature matrix ───────────────────────────
IDENTIFIER_COLS = [
    "session_id", "step_number", "candidate_item_id",
    "user_id", "restaurant_id", "label_addon_added", "_group_key",
]

LOW_SIGNAL_COLS = [
    # User segment dummies — all have ~4.8% positive rate (no discriminative power)
    "user_segment_premium", "user_segment_budget",
    "user_segment_frequent_high_value", "user_segment_regular",
    # Cuisine type dummies — near-zero correlation with label
    "cuisine_type_Chinese", "cuisine_type_Continental",
    "cuisine_type_Fast Food", "cuisine_type_Hyderabadi",
    "cuisine_type_Mughlai", "cuisine_type_North Indian",
    "cuisine_type_South Indian", "cuisine_type_Street Food",
    # Restaurant/time features — near-zero correlation
    "is_chain", "order_volume_30d", "hour_of_day",
    # Price bucket dummies — redundant with continuous price
    "price_bucket_high", "price_bucket_low",
    "price_bucket_medium", "price_bucket_premium",
    # String / object columns
    "city", "city_rest", "dish_subtype", "meal_time_bucket",
]

ITEM_CATEGORIES = ["main", "beverage", "dessert", "side"]


class FeaturePipeline:
    """
    Builds the full CSAO feature matrix from raw DataFrames.

    Parameters
    ----------
    items       : items_clean DataFrame
    users       : users DataFrame
    sessions    : sessions_clean DataFrame
    cart        : cart_events_clean DataFrame
    restaurants : restaurants DataFrame
    """

    def __init__(
        self,
        items:       pd.DataFrame,
        users:       pd.DataFrame,
        sessions:    pd.DataFrame,
        cart:        pd.DataFrame,
        restaurants: pd.DataFrame,
    ):
        self.items       = items.copy()
        self.users       = users.copy()
        self.sessions    = sessions.copy()
        self.cart        = cart.copy()
        self.restaurants = restaurants.copy()

        # Pre-compute lookup tables once
        self._build_cart_state_table()
        self._build_session_features()
        self._build_item_features()
        self._build_user_features()
        self._build_restaurant_features()

    # ── public ─────────────────────────────────────────────────────────────────

    def build(
        self,
        rank_df:   pd.DataFrame,
        split:     str = "train",
        reference: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build feature table for a rank DataFrame.

        Parameters
        ----------
        rank_df   : DataFrame with columns [session_id, step_number,
                    candidate_item_id, retrieval_score, src_*, label_addon_added]
        split     : "train" or "test"
        reference : train feature DataFrame; required when split="test"
                    to align column schema.

        Returns
        -------
        pd.DataFrame  — feature matrix ready for LightGBM
        """
        logger.info("[%s] Building features — %d rows", split, len(rank_df))
        df = rank_df.copy()

        # Standardise types
        for col in ["session_id", "step_number"]:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Join all feature tables
        df = df.merge(self._sess_feats,  on="session_id",        how="left")
        df = df.merge(self._cart_state,  on=["session_id", "step_number"], how="left")
        df = df.merge(self._item_feats,  on="candidate_item_id", how="left", suffixes=("", "_cand"))
        df = df.merge(self._user_feats,  on="user_id",           how="left", suffixes=("", "_user"))
        df = df.merge(self._rest_feats,  on="restaurant_id",     how="left", suffixes=("", "_rest"))

        # Resolve duplicate key columns from merges
        df = self._resolve_duplicates(df)

        # Step decay
        df["step_decay"] = 1.0 / df["step_number"].clip(lower=1)

        # Missing-meal flags
        for cat in ["beverage", "dessert", "side"]:
            df[f"missing_{cat}_flag"] = 1 - df[f"cart_has_{cat}"].fillna(0)

        # One-hot encode candidate category
        if "effective_category" in df.columns:
            for cat in ITEM_CATEGORIES:
                df[f"effective_category_{cat}"] = (
                    df["effective_category"] == cat
                ).astype(int)

        # Last-item category flags
        for cat in ITEM_CATEGORIES:
            df[f"last_item_category_{cat}"] = (
                df.get("last_item_category", pd.Series("unknown", index=df.index)) == cat
            ).astype(int)

        # Interaction features
        df = self._add_interactions(df)

        # Fill remaining NaNs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Drop non-model columns
        object_cols = df.select_dtypes(include="object").columns.tolist()
        drop = list(set(IDENTIFIER_COLS + LOW_SIGNAL_COLS + object_cols))
        drop = [c for c in drop if c in df.columns]
        df.drop(columns=drop, inplace=True)

        # Align schema to reference (test must match train)
        if reference is not None and split == "test":
            df = self._align_to_reference(df, reference)

        logger.info(
            "[%s] Feature table ready — shape: %s  features: %d",
            split, df.shape, len([c for c in df.columns if c not in IDENTIFIER_COLS]),
        )
        return df

    def get_feature_cols(self, feat_df: pd.DataFrame) -> list[str]:
        """Return model-ready feature column names (excluding identifiers/label)."""
        exclude = set(IDENTIFIER_COLS + ["label_addon_added"])
        return [c for c in feat_df.columns if c not in exclude]

    # ── pre-compute lookup tables ───────────────────────────────────────────────

    def _build_cart_state_table(self) -> None:
        """Cart state = cumulative cart composition at each (session, step)."""
        cart = self.cart.copy()
        cart = cart.merge(
            self.items[["item_id", "price", "effective_category"]],
            on="item_id", how="left",
        )
        cart = cart.sort_values(["session_id", "step_number"])

        cart["cart_item_count"] = cart.groupby("session_id").cumcount() + 1
        cart["cart_total_value"] = cart.groupby("session_id")["price"].cumsum()

        for cat in ITEM_CATEGORIES:
            cart[f"is_{cat}_item"] = (cart["effective_category"] == cat).astype(int)
            cart[f"cart_has_{cat}"] = cart.groupby("session_id")[f"is_{cat}_item"].cummax()

        cart["last_item_category"] = cart["effective_category"].fillna("unknown")

        self._cart_state = cart[[
            "session_id", "step_number",
            "cart_item_count", "cart_total_value",
            "cart_has_main", "cart_has_beverage", "cart_has_dessert", "cart_has_side",
            "last_item_category",
        ]].drop_duplicates(subset=["session_id", "step_number"])

    def _build_session_features(self) -> None:
        sess = self.sessions.copy()
        keep = ["session_id", "user_id", "restaurant_id",
                "meal_time_bucket", "is_weekend", "hour_of_day", "city"]
        keep = [c for c in keep if c in sess.columns]
        self._sess_feats = sess[keep].drop_duplicates(subset=["session_id"])

    def _build_item_features(self) -> None:
        items = self.items.copy()
        keep = [
            "item_id", "effective_category", "price",
            "popularity_score", "historical_attach_rate",
            "aggregate_rating", "overall_attach_rate",
            "meal_time_specificity", "meal_time_overlap",
        ]
        keep = [c for c in keep if c in items.columns]
        self._item_feats = (
            items[keep]
            .rename(columns={"item_id": "candidate_item_id"})
            .drop_duplicates(subset=["candidate_item_id"])
        )

    def _build_user_features(self) -> None:
        users = self.users.copy()
        keep = [
            "user_id", "user_segment",
            "order_count_90d", "recency_days", "avg_order_value",
            "veg_preference_ratio", "dessert_affinity_score",
            "beverage_affinity_score", "price_sensitivity_score",
        ]
        keep = [c for c in keep if c in users.columns]
        self._user_feats = users[keep].drop_duplicates(subset=["user_id"])

    def _build_restaurant_features(self) -> None:
        rests = self.restaurants.copy()
        keep = ["restaurant_id", "is_chain", "order_volume_30d", "city", "cuisine_type"]
        keep = [c for c in keep if c in rests.columns]
        self._rest_feats = rests[keep].drop_duplicates(subset=["restaurant_id"])

    # ── feature engineering helpers ─────────────────────────────────────────────

    @staticmethod
    def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """Add multiplicative interaction features."""

        def safe_mul(a: str, b: str, out: str) -> None:
            if a in df.columns and b in df.columns:
                df[out] = df[a].fillna(0) * df[b].fillna(0)

        # Retrieval × item quality signals
        safe_mul("retrieval_score", "historical_attach_rate", "retrieval_x_attach")
        safe_mul("retrieval_score", "popularity_score",       "retrieval_x_popularity")

        # Missing-meal × affinity signals  (key sequential signals)
        safe_mul("missing_beverage_flag", "beverage_affinity_score", "x_missing_bev_aff")
        safe_mul("missing_dessert_flag",  "dessert_affinity_score",  "x_missing_des_aff")
        safe_mul("missing_beverage_flag", "retrieval_score",         "missing_bev_x_retrieval")
        safe_mul("missing_dessert_flag",  "retrieval_score",         "missing_des_x_retrieval")

        # Price sensitivity × candidate price
        safe_mul("price_sensitivity_score", "price",            "x_price_sens_price")
        safe_mul("cart_total_value",        "price",            "x_cart_value_price")

        # Candidate category affinity
        safe_mul("effective_category_beverage", "beverage_affinity_score", "x_cand_bev_aff")
        safe_mul("effective_category_dessert",  "dessert_affinity_score",  "x_cand_des_aff")

        return df

    @staticmethod
    def _resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Clean up _x / _y suffix columns produced by merges."""
        for base in ["restaurant_id", "meal_time_bucket", "city"]:
            if f"{base}_x" in df.columns:
                df[base] = df[f"{base}_x"]
            for suf in ["_x", "_y"]:
                col = f"{base}{suf}"
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
        return df

    @staticmethod
    def _align_to_reference(
        df: pd.DataFrame, reference: pd.DataFrame
    ) -> pd.DataFrame:
        """Align test columns to match training schema exactly."""
        ref_cols = reference.columns.tolist()

        missing = [c for c in ref_cols if c not in df.columns]
        if missing:
            logger.warning("Adding %d missing columns to test (fill 0): %s", len(missing), missing)
            for c in missing:
                df[c] = 0

        extra = [c for c in df.columns if c not in ref_cols]
        if extra:
            logger.warning("Dropping %d extra columns from test: %s", len(extra), extra)
            df.drop(columns=extra, inplace=True)

        return df[ref_cols]
