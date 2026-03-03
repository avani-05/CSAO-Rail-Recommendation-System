"""
src/data_loader.py
==================
Load, validate, and type-cast all raw input data files for the
CartNext CSAO Rail Recommendation System.

Usage
-----
    from src.data_loader import CSAODataLoader
    loader = CSAODataLoader(base_dir="data/raw")
    data   = loader.load_all()
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Expected schema: (dtype, nullable) ─────────────────────────────────────────
_ITEM_SCHEMA = {
    "item_id":              ("int64",   False),
    "restaurant_id":        ("int64",   False),
    "effective_category":   ("object",  False),
    "price":                ("float64", False),
    "popularity_score":     ("float64", True),
    "historical_attach_rate": ("float64", True),
    "aggregate_rating":     ("float64", True),
    "overall_attach_rate":  ("float64", True),
    "meal_time_specificity":("float64", True),
    "meal_time_overlap":    ("float64", True),
}

_USER_SCHEMA = {
    "user_id":               ("int64",   False),
    "user_segment":          ("object",  True),
    "order_count_90d":       ("float64", True),
    "recency_days":          ("float64", True),
    "avg_order_value":       ("float64", True),
    "veg_preference_ratio":  ("float64", True),
    "dessert_affinity_score":("float64", True),
    "beverage_affinity_score":("float64",True),
    "price_sensitivity_score":("float64",True),
}

_SESSION_SCHEMA = {
    "session_id":       ("int64",  False),
    "user_id":          ("int64",  True),
    "restaurant_id":    ("int64",  False),
    "meal_time_bucket": ("object", True),
    "is_weekend":       ("int64",  True),
    "hour_of_day":      ("int64",  True),
    "city":             ("object", True),
}

_CART_SCHEMA = {
    "session_id":  ("int64", False),
    "step_number": ("int64", False),
    "item_id":     ("int64", False),
}

_RESTAURANT_SCHEMA = {
    "restaurant_id":   ("int64",   False),
    "is_chain":        ("int64",   True),
    "order_volume_30d":("float64", True),
    "city":            ("object",  True),
    "cuisine_type":    ("object",  True),
}

_RANK_SCHEMA = {
    "session_id":       ("int64",   False),
    "step_number":      ("int64",   False),
    "candidate_item_id":("int64",   False),
    "retrieval_score":  ("float64", False),
    "src_cooc":         ("int64",   True),
    "src_ctx":          ("int64",   True),
    "src_rule":         ("int64",   True),
    "label_addon_added":("int64",   False),
}


@dataclass
class CSAODataset:
    """Container for all loaded DataFrames."""
    items:      pd.DataFrame = field(default_factory=pd.DataFrame)
    users:      pd.DataFrame = field(default_factory=pd.DataFrame)
    sessions:   pd.DataFrame = field(default_factory=pd.DataFrame)
    cart:       pd.DataFrame = field(default_factory=pd.DataFrame)
    restaurants:pd.DataFrame = field(default_factory=pd.DataFrame)
    rank_train: pd.DataFrame = field(default_factory=pd.DataFrame)
    rank_test:  pd.DataFrame = field(default_factory=pd.DataFrame)


class CSAODataLoader:
    """
    Loads all CSV data files, enforces schema, and returns a CSAODataset.

    Parameters
    ----------
    base_dir : str | Path
        Root directory containing all CSV files.
    strict : bool
        If True, raise on schema violations. If False, warn and continue.
    """

    _FILE_MAP = {
        "items":       "items_clean.csv",
        "users":       "users.csv",
        "sessions":    "sessions_clean.csv",
        "cart":        "cart_events_clean.csv",
        "restaurants": "restaurants.csv",
        "rank_train":  "rank_train_data.csv",
        "rank_test":   "rank_test_data.csv",
    }

    def __init__(self, base_dir: str | Path = "data/raw", strict: bool = False):
        self.base_dir = Path(base_dir)
        self.strict   = strict

    # ── public ─────────────────────────────────────────────────────────────────

    def load_all(self) -> CSAODataset:
        """Load all files and return a validated CSAODataset."""
        logger.info("Loading CSAO dataset from %s", self.base_dir)
        ds = CSAODataset(
            items       = self._load("items",       _ITEM_SCHEMA),
            users       = self._load("users",       _USER_SCHEMA),
            sessions    = self._load("sessions",    _SESSION_SCHEMA),
            cart        = self._load("cart",        _CART_SCHEMA),
            restaurants = self._load("restaurants", _RESTAURANT_SCHEMA),
            rank_train  = self._load("rank_train",  _RANK_SCHEMA),
            rank_test   = self._load("rank_test",   _RANK_SCHEMA),
        )
        self._post_validate(ds)
        self._log_summary(ds)
        return ds

    def load_features(
        self,
        train_path: str | Path = "data/model_input/lgbm_train_features_v2.csv",
        test_path:  str | Path = "data/model_input/lgbm_test_features_v2.csv",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load pre-built feature tables (output of feature_pipeline.py)."""
        train_path = Path(train_path)
        test_path  = Path(test_path)

        logger.info("Loading train features from %s", train_path)
        train_df = pd.read_csv(train_path)

        logger.info("Loading test features from %s", test_path)
        test_df = pd.read_csv(test_path)

        # Align columns: test must match train exactly
        train_df, test_df = self._align_columns(train_df, test_df)

        logger.info(
            "Feature tables loaded — train: %s  test: %s",
            train_df.shape, test_df.shape,
        )
        return train_df, test_df

    # ── private ─────────────────────────────────────────────────────────────────

    def _load(self, key: str, schema: dict) -> pd.DataFrame:
        path = self.base_dir / self._FILE_MAP[key]
        if not path.exists():
            msg = f"File not found: {path}"
            if self.strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)
            return pd.DataFrame()

        df = pd.read_csv(path)
        df = self._normalise_category(df, key)
        df = self._cast_schema(df, schema, key)
        return df

    def _normalise_category(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """Rename normalized_category → effective_category if needed."""
        if key == "items" and "effective_category" not in df.columns:
            if "normalized_category" in df.columns:
                df = df.rename(columns={"normalized_category": "effective_category"})
                logger.debug("items: renamed normalized_category → effective_category")
        return df

    def _cast_schema(
        self, df: pd.DataFrame, schema: dict, name: str
    ) -> pd.DataFrame:
        """Type-cast columns to expected dtypes; warn on missing columns."""
        for col, (dtype, nullable) in schema.items():
            if col not in df.columns:
                msg = f"[{name}] Missing column: {col}"
                if self.strict:
                    raise KeyError(msg)
                logger.warning(msg)
                continue

            try:
                if dtype in ("int64", "float64"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if dtype == "int64" and nullable is False:
                        df[col] = df[col].fillna(0).astype("int64")
                    elif dtype == "int64":
                        df[col] = df[col].astype("Int64")  # nullable int
                    else:
                        df[col] = df[col].astype("float64")
                else:
                    df[col] = df[col].astype("object")
            except Exception as exc:
                logger.warning("[%s] Could not cast %s to %s: %s", name, col, dtype, exc)
        return df

    @staticmethod
    def _align_columns(
        train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Make test columns match train exactly:
        - Add columns present in train but missing from test (fill 0)
        - Drop columns in test that are absent from train
        - Reorder test to match train
        """
        train_cols = train.columns.tolist()
        test_cols  = test.columns.tolist()

        missing = [c for c in train_cols if c not in test_cols]
        if missing:
            logger.warning("Adding %d missing columns to test (filling 0): %s", len(missing), missing)
            for c in missing:
                test[c] = 0

        extra = [c for c in test.columns if c not in train_cols]
        if extra:
            logger.warning("Dropping %d extra columns from test: %s", len(extra), extra)
            test = test.drop(columns=extra)

        test = test[train_cols]
        return train, test

    @staticmethod
    def _post_validate(ds: CSAODataset) -> None:
        """Cross-table sanity checks."""
        if ds.rank_train.empty or ds.rank_test.empty:
            return

        # No session overlap between train and test
        train_sids = set(ds.rank_train["session_id"].unique())
        test_sids  = set(ds.rank_test["session_id"].unique())
        overlap    = train_sids & test_sids
        if overlap:
            logger.warning(
                "Data leakage: %d session_ids appear in both train and test!", len(overlap)
            )

        # Every group must have at least 1 positive label
        for split, df in [("train", ds.rank_train), ("test", ds.rank_test)]:
            pos_per_group = (
                df.groupby(["session_id", "step_number"])["label_addon_added"].sum()
            )
            zero_pos = (pos_per_group == 0).sum()
            if zero_pos > 0:
                logger.warning(
                    "[%s] %d groups have zero positive labels — these will be excluded from metrics",
                    split, zero_pos,
                )

        # Positive rate sanity
        for split, df in [("train", ds.rank_train), ("test", ds.rank_test)]:
            rate = df["label_addon_added"].mean()
            if rate < 0.01 or rate > 0.20:
                logger.warning(
                    "[%s] Unusual positive rate: %.4f (expected 0.03–0.06)", split, rate
                )

    @staticmethod
    def _log_summary(ds: CSAODataset) -> None:
        rows = {
            "items":       len(ds.items),
            "users":       len(ds.users),
            "sessions":    len(ds.sessions),
            "cart":        len(ds.cart),
            "restaurants": len(ds.restaurants),
            "rank_train":  len(ds.rank_train),
            "rank_test":   len(ds.rank_test),
        }
        logger.info("Dataset summary:")
        for k, n in rows.items():
            logger.info("  %-16s %10d rows", k, n)

        if not ds.rank_train.empty:
            logger.info(
                "  Train positive rate : %.4f", ds.rank_train["label_addon_added"].mean()
            )
        if not ds.rank_test.empty:
            logger.info(
                "  Test  positive rate : %.4f", ds.rank_test["label_addon_added"].mean()
            )


# ── convenience function ────────────────────────────────────────────────────────

def load_dataset(base_dir: str = "data/raw", strict: bool = False) -> CSAODataset:
    """One-liner helper: load everything with default paths."""
    return CSAODataLoader(base_dir=base_dir, strict=strict).load_all()


def load_feature_tables(
    train_path: str = "data/model_input/lgbm_train_features_v2.csv",
    test_path:  str = "data/model_input/lgbm_test_features_v2.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-liner helper: load pre-built feature CSVs."""
    return CSAODataLoader().load_features(train_path, test_path)
