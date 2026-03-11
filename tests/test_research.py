"""
Tests for the research / ML pipeline.

Covers: targets, feature engineering, dataset construction, preprocessing,
model training, evaluation, walk-forward splits, and external providers.
"""

from __future__ import annotations

import asyncio
import math
import numpy as np
import pandas as pd
import pytest

from app.research.targets import (
    add_direction_target,
    add_edge_target,
    add_all_targets,
)
from app.research.feature_eng import (
    engineer_features,
    get_ml_feature_names,
    RAW_FEATURE_COLS,
    DERIVED_FEATURE_COLS,
)
from app.research.dataset import (
    generate_synthetic_dataset,
    train_test_split_temporal,
    walk_forward_splits,
    SplitIndices,
)
from app.research.preprocessing import (
    build_preprocessing_pipeline,
    prepare_X,
    WinsorizeTransformer,
)
from app.research.models import (
    train_single_model,
    train_all_models,
    save_model_artifact,
    load_model_artifact,
    calibrate_model,
)
from app.research.evaluation import (
    evaluate_classifier,
    simulate_profit,
    walk_forward_evaluate,
)
from app.research.report import build_training_report, format_text_summary


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_df(n: int = 200) -> pd.DataFrame:
    """Minimal DataFrame with required columns for testing."""
    return generate_synthetic_dataset(n=n, seed=42)


# ── Targets ────────────────────────────────────────────────────────────────


class TestTargets:
    def test_direction_target_shape(self):
        df = _make_df(100)
        result = add_direction_target(df, horizon=5, min_edge=0.001)
        assert "target_direction" in result.columns
        valid = result["target_direction"].dropna()
        assert len(valid) == 95  # last 5 are NaN
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_edge_target_is_continuous(self):
        df = _make_df(100)
        result = add_edge_target(df, horizon=5, fee=0.02)
        assert "target_edge" in result.columns
        valid = result["target_edge"].dropna()
        assert len(valid) > 0
        assert valid.dtype == np.float64

    def test_both_targets_added(self):
        df = _make_df(100)
        result = add_all_targets(df, horizon=5)
        assert "target_direction" in result.columns
        assert "target_edge" in result.columns

    def test_no_future_leakage_direction(self):
        """The target should be NaN for the last `horizon` rows."""
        df = _make_df(50)
        result = add_direction_target(df, horizon=3, min_edge=0.001)
        tail = result["target_direction"].iloc[-3:]
        assert tail.isna().all()


# ── Feature Engineering ────────────────────────────────────────────────────


class TestFeatureEngineering:
    def test_derived_columns_added(self):
        df = _make_df(50)
        result = engineer_features(df)
        for col in DERIVED_FEATURE_COLS:
            assert col in result.columns, f"Missing derived column: {col}"

    def test_spread_pct_computed(self):
        df = _make_df(50)
        result = engineer_features(df)
        non_null = result["spread_pct"].dropna()
        assert len(non_null) > 0
        assert (non_null >= 0).all()

    def test_missing_columns_filled(self):
        df = pd.DataFrame({"mid_price": [0.5, 0.6], "spread": [0.02, 0.03]})
        result = engineer_features(df)
        assert "bid_depth_5c" in result.columns

    def test_feature_names_list(self):
        names = get_ml_feature_names(include_external=False)
        assert len(names) == len(RAW_FEATURE_COLS) + len(DERIVED_FEATURE_COLS)

        with_ext = get_ml_feature_names(include_external=True)
        assert len(with_ext) > len(names)


# ── Dataset / Splitting ───────────────────────────────────────────────────


class TestDataset:
    def test_synthetic_dataset_shape(self):
        df = generate_synthetic_dataset(n=500)
        assert len(df) == 500
        assert "mid_price" in df.columns
        assert "spread" in df.columns

    def test_temporal_split_no_overlap(self):
        train_idx, test_idx = train_test_split_temporal(100, test_frac=0.2)
        assert len(train_idx) == 80
        assert len(test_idx) == 20
        assert train_idx.max() < test_idx.min()

    def test_walk_forward_no_leakage(self):
        folds = walk_forward_splits(500, n_folds=4, min_train_frac=0.3)
        assert len(folds) >= 2

        for fold in folds:
            assert fold.train_end <= fold.val_start
            assert fold.val_start < fold.val_end

        for i in range(1, len(folds)):
            assert folds[i].train_end >= folds[i - 1].train_end

    def test_walk_forward_expanding_window(self):
        """Train sets should grow across folds."""
        folds = walk_forward_splits(1000, n_folds=5)
        train_sizes = [f.train_end - f.train_start for f in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]


# ── Preprocessing ─────────────────────────────────────────────────────────


class TestPreprocessing:
    def test_winsorize(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        X[0, 0] = 100.0  # outlier
        w = WinsorizeTransformer(lower=5, upper=95)
        Xt = w.fit_transform(X)
        assert Xt[0, 0] < 100.0

    def test_pipeline_handles_nan(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
        pipe = build_preprocessing_pipeline(["a", "b"], scale=True)
        Xt = pipe.fit_transform(X)
        assert not np.isnan(Xt).any()

    def test_prepare_X_fills_missing_cols(self):
        df = pd.DataFrame({"spread": [0.02], "mid_price": [0.5]})
        X, names = prepare_X(df, feature_names=["spread", "mid_price", "nonexistent"])
        assert X.shape == (1, 3)
        assert np.isnan(X[0, 2])


# ── Model Training ────────────────────────────────────────────────────────


class TestModelTraining:
    @pytest.fixture
    def train_data(self):
        df = generate_synthetic_dataset(n=2000, seed=42)
        df = engineer_features(df)
        df = add_all_targets(df, horizon=5, min_edge=0.005)
        df = df.dropna(subset=["target_direction"]).reset_index(drop=True)
        feature_names = get_ml_feature_names()
        X, _ = prepare_X(df, feature_names)
        y = df["target_direction"].values.astype(int)
        # Ensure both classes are present in train and val
        assert y.sum() > 0, "No positive labels — adjust min_edge or data gen"
        assert (y == 0).sum() > 0, "No negative labels"
        split = int(len(X) * 0.7)
        return X[:split], y[:split], X[split:], y[split:], feature_names

    def test_train_logistic(self, train_data):
        X_tr, y_tr, X_val, y_val, names = train_data
        m = train_single_model("logistic_regression", X_tr, y_tr, X_val, y_val, names)
        assert m.name == "logistic_regression"
        assert 0 < m.val_log_loss < 10
        assert 0 < m.val_accuracy <= 1

    def test_train_gradient_boosting(self, train_data):
        X_tr, y_tr, X_val, y_val, names = train_data
        m = train_single_model("gradient_boosting", X_tr, y_tr, X_val, y_val, names)
        assert m.name == "gradient_boosting"

    def test_train_random_forest(self, train_data):
        X_tr, y_tr, X_val, y_val, names = train_data
        m = train_single_model("random_forest", X_tr, y_tr, X_val, y_val, names)
        assert m.name == "random_forest"

    def test_train_all_sorted(self, train_data):
        X_tr, y_tr, X_val, y_val, names = train_data
        models = train_all_models(X_tr, y_tr, X_val, y_val, names)
        assert len(models) == 3
        assert models[0].val_log_loss <= models[-1].val_log_loss

    def test_calibrate_model(self, train_data):
        X_tr, y_tr, X_val, y_val, names = train_data
        m = train_single_model("logistic_regression", X_tr, y_tr, X_val, y_val, names)
        m = calibrate_model(m, X_val, y_val)
        X_val_t = m.pipeline.transform(X_val)
        proba = m.classifier.predict_proba(X_val_t)
        assert proba.shape[1] == 2
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_and_load_artifact(self, train_data, tmp_path):
        X_tr, y_tr, X_val, y_val, names = train_data
        m = train_single_model("logistic_regression", X_tr, y_tr, X_val, y_val, names)
        path = save_model_artifact(m, names, tmp_path)
        assert path.exists()

        loaded = load_model_artifact(path)
        assert loaded["name"] == "logistic_regression"
        assert loaded["feature_names"] == names
        assert loaded["pipeline"] is not None


# ── Evaluation ────────────────────────────────────────────────────────────


class TestEvaluation:
    def test_evaluate_classifier(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_proba = np.array([
            [0.8, 0.2], [0.7, 0.3], [0.3, 0.7],
            [0.4, 0.6], [0.9, 0.1], [0.2, 0.8],
        ])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        result = evaluate_classifier(y_true, y_proba, y_pred)
        assert result.accuracy == 1.0
        assert result.log_loss_val > 0
        assert result.confusion is not None

    def test_simulate_profit_no_trades(self):
        y_true = np.array([0, 1, 0])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.95, 0.05]])
        entry = np.array([0.5, 0.5, 0.5])
        exit_ = np.array([0.48, 0.53, 0.49])
        result = simulate_profit(y_true, y_proba, entry, exit_, threshold=0.9)
        assert result["n_trades"] == 0
        assert result["total_pnl"] == 0.0

    def test_simulate_profit_with_trades(self):
        y_true = np.array([1, 0, 1])
        y_proba = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
        entry = np.array([0.52, 0.52, 0.52])
        exit_ = np.array([0.55, 0.48, 0.56])
        result = simulate_profit(y_true, y_proba, entry, exit_, threshold=0.6, fee=0.01)
        assert result["n_trades"] == 3
        assert isinstance(result["total_pnl"], float)
        assert 0 <= result["win_rate"] <= 1

    def test_walk_forward_evaluate(self):
        from app.research.evaluation import EvaluationResult

        results = [
            EvaluationResult(accuracy=0.6, log_loss_val=0.65, roc_auc=0.55),
            EvaluationResult(accuracy=0.7, log_loss_val=0.55, roc_auc=0.65),
        ]
        summary = walk_forward_evaluate(results)
        assert summary["n_folds"] == 2
        assert 0.6 <= summary["mean_accuracy"] <= 0.7


# ── Report ────────────────────────────────────────────────────────────────


class TestReport:
    def test_build_report_structure(self):
        from app.research.evaluation import EvaluationResult

        er = EvaluationResult(accuracy=0.65, log_loss_val=0.60, roc_auc=0.62, brier_score=0.25)
        report = build_training_report(
            model_name="test_model",
            feature_names=["a", "b", "c"],
            train_size=100,
            val_size=20,
            test_size=30,
            eval_result=er,
        )
        assert report["model_name"] == "test_model"
        assert report["dataset"]["total"] == 150
        assert "analysis" in report
        assert "overfitting_risks" in report["analysis"]

    def test_text_summary_renders(self):
        from app.research.evaluation import EvaluationResult

        er = EvaluationResult(accuracy=0.65, log_loss_val=0.60, brier_score=0.25)
        report = build_training_report(
            model_name="test",
            feature_names=["x"],
            train_size=50,
            val_size=10,
            test_size=10,
            eval_result=er,
        )
        text = format_text_summary(report)
        assert "MODEL TRAINING REPORT" in text
        assert "Accuracy" in text


# ── External Providers ────────────────────────────────────────────────────


class TestProviders:
    def test_sentiment_provider_unavailable(self):
        from app.data.providers.sentiment import SentimentProvider
        p = SentimentProvider()
        assert p.is_available() is False
        assert p.feature_names() == ["sentiment_score"]

    def test_news_provider_unavailable(self):
        from app.data.providers.news import NewsProvider
        p = NewsProvider()
        assert p.is_available() is False
        assert p.feature_names() == ["news_intensity"]

    @pytest.mark.asyncio
    async def test_sentiment_returns_nan_when_unavailable(self):
        from app.data.providers.sentiment import SentimentProvider
        p = SentimentProvider()
        result = await p.fetch_features("market1", "token1")
        assert "sentiment_score" in result
        assert math.isnan(result["sentiment_score"])

    @pytest.mark.asyncio
    async def test_news_returns_nan_when_unavailable(self):
        from app.data.providers.news import NewsProvider
        p = NewsProvider()
        result = await p.fetch_features("market1", "token1")
        assert "news_intensity" in result
        assert math.isnan(result["news_intensity"])

    def test_registry_lists_providers(self):
        from app.data.providers.base import ProviderRegistry
        available = ProviderRegistry.available()
        assert "sentiment" in available
        assert "news" in available

    @pytest.mark.asyncio
    async def test_registry_fetch_all_no_crash(self):
        from app.data.providers.base import ProviderRegistry
        result = await ProviderRegistry.fetch_all("m", "t")
        assert isinstance(result, dict)
