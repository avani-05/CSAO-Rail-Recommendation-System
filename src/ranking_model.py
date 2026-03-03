"""
src/ranking_model.py
=====================
LightGBM binary classifier used as the Stage-2 CSAO ranker.

Model rationale
---------------
With ~25 candidates per query group and ~3.8% positive rate (1 positive
per group), pairwise ranking objectives (LambdaRank, rank_xendcg) produce
near-zero gradients. Binary cross-entropy handles this naturally — the
predicted probability is a valid ranking score at inference time.
This is the same approach used in production at DoorDash, Instacart,
and Zomato.

Usage
-----
    from src.ranking_model import CSAORanker

    # Train
    ranker = CSAORanker()
    ranker.fit(train_df, FEATURE_COLS)

    # Evaluate
    scores = ranker.predict(test_df)

    # Save / load
    ranker.save("models/csao_ranker.lgb")
    ranker = CSAORanker.load("models/csao_ranker.lgb")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default hyperparameters (mirror notebook Cell 4) ───────────────────────────
DEFAULT_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "num_leaves":       63,
    "learning_rate":    0.05,
    "lambda_l1":        0.0,
    "lambda_l2":        0.0,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "feature_fraction": 0.8,
    "min_data_in_leaf": 20,
    "max_bin":          255,
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             42,
}

DEFAULT_TRAIN_CONFIG = {
    "num_boost_round":       3000,
    "early_stopping_rounds": 100,
    "verbose_eval":          50,
}


class CSAORanker:
    """
    LightGBM binary ranker for CSAO rail recommendation.

    Parameters
    ----------
    params       : LightGBM training parameters (overrides DEFAULT_PARAMS)
    train_config : Training loop configuration (num_boost_round, etc.)
    """

    def __init__(
        self,
        params:       Optional[dict] = None,
        train_config: Optional[dict] = None,
    ):
        try:
            import lightgbm as lgb
            self._lgb = lgb
        except ImportError:
            raise ImportError("lightgbm is required: pip install lightgbm")

        self.params       = {**DEFAULT_PARAMS,       **(params or {})}
        self.train_config = {**DEFAULT_TRAIN_CONFIG, **(train_config or {})}
        self.model        = None
        self.feature_cols: List[str] = []
        self._train_info: dict = {}

    # ── training ────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df:     pd.DataFrame,
        feature_cols: List[str],
        test_df:      Optional[pd.DataFrame] = None,
        label_col:    str = "label_addon_added",
    ) -> "CSAORanker":
        """
        Train the LightGBM binary ranker.

        The positive-class weight (scale_pos_weight) is computed automatically
        from the label distribution: neg_count / pos_count.

        Parameters
        ----------
        train_df     : feature table with label column
        feature_cols : list of feature column names
        test_df      : optional holdout for early stopping
        label_col    : name of the binary label column

        Returns
        -------
        self
        """
        self.feature_cols = list(feature_cols)

        # Sort by group key for consistent LightGBM dataset construction
        train_sorted = train_df.sort_values(
            ["session_id", "step_number"]
        ).reset_index(drop=True)

        X_train = train_sorted[feature_cols].values.astype(np.float32)
        y_train = train_sorted[label_col].values.astype(int)

        # Auto-compute class weight
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos
        logger.info(
            "Train: %d neg, %d pos | scale_pos_weight = %.2f",
            neg, pos, scale_pos_weight,
        )
        self.params["scale_pos_weight"] = scale_pos_weight

        lgb_train = self._lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_cols,
            free_raw_data=False,
        )

        callbacks = []
        valid_sets = [lgb_train]
        valid_names = ["train"]
        evals_result: dict = {}

        if test_df is not None:
            test_sorted = test_df.sort_values(
                ["session_id", "step_number"]
            ).reset_index(drop=True)
            X_test = test_sorted[feature_cols].values.astype(np.float32)
            y_test = test_sorted[label_col].values.astype(int)

            lgb_test = self._lgb.Dataset(
                X_test,
                label=y_test,
                feature_name=self.feature_cols,
                reference=lgb_train,
                free_raw_data=False,
            )
            valid_sets  = [lgb_train, lgb_test]
            valid_names = ["train", "test"]
            callbacks.append(
                self._lgb.early_stopping(
                    self.train_config["early_stopping_rounds"], verbose=True
                )
            )

        callbacks.append(
            self._lgb.log_evaluation(
                period=self.train_config["verbose_eval"]
            )
        )

        logger.info("Starting LightGBM training ...")
        t0 = time.perf_counter()

        self.model = self._lgb.train(
            params        = self.params,
            train_set     = lgb_train,
            num_boost_round = self.train_config["num_boost_round"],
            valid_sets    = valid_sets,
            valid_names   = valid_names,
            callbacks     = callbacks,
            evals_result  = evals_result,
        )

        elapsed = time.perf_counter() - t0
        self._train_info = {
            "best_iteration":  self.model.best_iteration,
            "best_score":      self.model.best_score,
            "num_trees":       self.model.num_trees(),
            "train_seconds":   elapsed,
            "n_features":      len(self.feature_cols),
            "scale_pos_weight": scale_pos_weight,
        }
        logger.info(
            "Training complete in %.1fs | best_iteration=%d | AUC(test)=%.5f",
            elapsed,
            self.model.best_iteration,
            self.model.best_score.get("test", {}).get("auc", float("nan")),
        )
        return self

    # ── inference ───────────────────────────────────────────────────────────────

    def predict(
        self,
        df:            pd.DataFrame,
        score_col:     str = "model_score",
        num_iteration: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Score candidates and return df with a new `score_col` column.

        Parameters
        ----------
        df            : feature DataFrame
        score_col     : name of the output score column
        num_iteration : if None, uses model.best_iteration

        Returns
        -------
        Input df with `score_col` appended (original row order preserved).
        """
        self._require_fitted()
        n_iter = num_iteration or self.model.best_iteration

        # Ensure all required features are present
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning("Adding %d missing feature columns (fill 0): %s", len(missing), missing)
            df = df.copy()
            for c in missing:
                df[c] = 0

        X = df[self.feature_cols].values.astype(np.float32)
        scores = self.model.predict(X, num_iteration=n_iter)
        out = df.copy()
        out[score_col] = scores
        return out

    def rank(
        self,
        df:       pd.DataFrame,
        group_col: str = "_group_key",
        top_k:    int  = 8,
    ) -> pd.DataFrame:
        """
        Score and return top-k candidates per group.

        Parameters
        ----------
        df        : feature DataFrame (must have group_col)
        group_col : column identifying the query group
        top_k     : number of top candidates to return per group

        Returns
        -------
        pd.DataFrame with top_k rows per group, sorted by model_score descending.
        """
        scored = self.predict(df)
        ranked = (
            scored
            .sort_values("model_score", ascending=False)
            .groupby(group_col, sort=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return ranked

    def benchmark_latency(
        self,
        df:          pd.DataFrame,
        n_candidates: int = 25,
        n_trials:    int = 1000,
    ) -> dict:
        """
        Measure inference latency for a single query of n_candidates rows.

        Returns dict with mean, p50, p95, p99, p999, max (all in ms).
        """
        self._require_fitted()
        X = df[self.feature_cols].head(n_candidates).values.astype(np.float32)

        # Warm-up
        for _ in range(10):
            self.model.predict(X)

        latencies = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            self.model.predict(X)
            latencies.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies)
        result = {
            "mean_ms": float(arr.mean()),
            "p50_ms":  float(np.percentile(arr, 50)),
            "p95_ms":  float(np.percentile(arr, 95)),
            "p99_ms":  float(np.percentile(arr, 99)),
            "p999_ms": float(np.percentile(arr, 99.9)),
            "max_ms":  float(arr.max()),
            "n_trials": n_trials,
            "n_candidates": n_candidates,
        }
        logger.info(
            "Latency benchmark | mean=%.3fms P50=%.3fms P95=%.3fms P99=%.3fms",
            result["mean_ms"], result["p50_ms"], result["p95_ms"], result["p99_ms"],
        )
        return result

    # ── feature importance ──────────────────────────────────────────────────────

    def feature_importance(
        self, importance_type: str = "gain", top_n: int = 20
    ) -> pd.DataFrame:
        """Return feature importance DataFrame sorted descending."""
        self._require_fitted()
        df = pd.DataFrame({
            "feature":    self.model.feature_name(),
            "importance": self.model.feature_importance(importance_type=importance_type),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["share_pct"] = df["importance"] / df["importance"].sum() * 100
        return df.head(top_n)

    # ── persistence ─────────────────────────────────────────────────────────────

    def save(self, path: str = "models/csao_ranker.lgb") -> None:
        """Save the trained model to a LightGBM text file."""
        self._require_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        logger.info("Model saved → %s  (trees: %d)", path, self.model.num_trees())

    @classmethod
    def load(cls, path: str = "models/csao_ranker.lgb") -> "CSAORanker":
        """Load a saved LightGBM model and return a CSAORanker instance."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required: pip install lightgbm")

        ranker = cls()
        ranker.model = lgb.Booster(model_file=path)
        ranker.feature_cols = ranker.model.feature_name()
        logger.info(
            "Model loaded from %s | trees: %d | features: %d",
            path,
            ranker.model.num_trees(),
            len(ranker.feature_cols),
        )
        return ranker

    # ── helpers ─────────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

    def __repr__(self) -> str:
        status = (
            f"trees={self.model.num_trees()}, features={len(self.feature_cols)}"
            if self.model else "not fitted"
        )
        return f"CSAORanker({status})"
