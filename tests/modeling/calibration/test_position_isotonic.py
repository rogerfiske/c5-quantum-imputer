"""Tests for position isotonic calibration module."""
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from src.modeling.calibration.position_isotonic import (
    CalibrationPackConfig,
    PositionIsotonicCalibrator,
    _hits_at_k,
    distribution_breakdown,
    recall_at_20,
)


class TestPositionIsotonicCalibrator:
    """Tests for PositionIsotonicCalibrator class."""

    @pytest.fixture
    def synthetic_scores(self):
        """Generate synthetic score matrix (n_events, 39)."""
        np.random.seed(42)
        return np.random.rand(100, 39)

    @pytest.fixture
    def synthetic_labels(self):
        """
        Generate synthetic binary labels (n_events, 39) with exactly 5 ones per row.

        This simulates lottery data where exactly 5 positions are winners per event.
        """
        np.random.seed(42)
        labels = np.zeros((100, 39), dtype=int)
        for i in range(100):
            labels[i, np.random.choice(39, 5, replace=False)] = 1
        return labels

    @pytest.fixture
    def calibrator(self, synthetic_scores, synthetic_labels):
        """Provide a fitted calibrator instance for tests."""
        cal = PositionIsotonicCalibrator()
        cal.fit(synthetic_scores, synthetic_labels)
        return cal

    def test_init_default_config(self):
        """Test that __init__ creates default config when none provided."""
        cal = PositionIsotonicCalibrator()
        assert isinstance(cal.config, CalibrationPackConfig)
        assert cal.config.normalize is True
        assert cal._fitted is False
        assert len(cal.models) == 0

    def test_init_custom_config(self):
        """Test that __init__ accepts custom config."""
        config = CalibrationPackConfig(normalize=False)
        cal = PositionIsotonicCalibrator(config=config)
        assert cal.config.normalize is False

    def test_fit_creates_39_models(self, synthetic_scores, synthetic_labels):
        """Test that fit creates one isotonic regressor per position."""
        cal = PositionIsotonicCalibrator()
        cal.fit(synthetic_scores, synthetic_labels)

        assert len(cal.models) == 39, f"Expected 39 models, got {len(cal.models)}"
        assert cal._fitted is True

        # Verify all models are IsotonicRegression instances
        from sklearn.isotonic import IsotonicRegression

        for j, model in cal.models.items():
            assert isinstance(model, IsotonicRegression)
            assert 0 <= j < 39

    def test_fit_returns_self(self, synthetic_scores, synthetic_labels):
        """Test that fit returns self for method chaining."""
        cal = PositionIsotonicCalibrator()
        result = cal.fit(synthetic_scores, synthetic_labels)
        assert result is cal

    def test_transform_output_shape(self, calibrator, synthetic_scores):
        """Test that transform returns correct shape."""
        probs = calibrator.transform(synthetic_scores)
        assert probs.shape == (100, 39), f"Expected shape (100, 39), got {probs.shape}"

    def test_transform_probabilities_sum_to_one(self, calibrator, synthetic_scores):
        """
        Test that probabilities are normalized per event.

        Each event's probabilities should sum to 1.0 within numerical tolerance.
        """
        probs = calibrator.transform(synthetic_scores)
        row_sums = probs.sum(axis=1)
        npt.assert_allclose(
            row_sums,
            1.0,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Row sums should be 1.0 (normalized probabilities)",
        )

    def test_transform_without_normalization(self, synthetic_scores, synthetic_labels):
        """Test transform without normalization produces raw calibrated scores."""
        config = CalibrationPackConfig(normalize=False)
        cal = PositionIsotonicCalibrator(config=config)
        cal.fit(synthetic_scores, synthetic_labels)
        probs = cal.transform(synthetic_scores)

        # Without normalization, sums may not be 1.0
        row_sums = probs.sum(axis=1)
        # At least some rows should not sum to exactly 1.0
        assert not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-6)

    def test_transform_raises_if_not_fitted(self, synthetic_scores):
        """Test that transform raises AssertionError if calibrator not fitted."""
        cal = PositionIsotonicCalibrator()
        with pytest.raises(AssertionError, match="Calibrator not fitted"):
            cal.transform(synthetic_scores)

    def test_transform_output_dtype_is_float(self, calibrator, synthetic_scores):
        """Test that transform output is float dtype."""
        probs = calibrator.transform(synthetic_scores)
        assert probs.dtype == np.float64

    def test_save_and_load_roundtrip(self, calibrator, synthetic_scores):
        """Test that save() and load() preserve calibrator state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "calibrator.joblib"

            # Save calibrator
            calibrator.save(str(save_path))
            assert save_path.exists()

            # Load calibrator
            loaded_cal = PositionIsotonicCalibrator.load(str(save_path))

            # Verify loaded calibrator has same config
            assert loaded_cal.config.normalize == calibrator.config.normalize
            assert loaded_cal._fitted is True
            assert len(loaded_cal.models) == 39

            # Verify loaded calibrator produces same outputs
            original_probs = calibrator.transform(synthetic_scores)
            loaded_probs = loaded_cal.transform(synthetic_scores)
            npt.assert_array_equal(
                original_probs,
                loaded_probs,
                err_msg="Loaded calibrator should produce identical outputs",
            )

    def test_calibration_reduces_extreme_scores(self, synthetic_labels):
        """
        Test that calibration deflates extreme uncalibrated scores.

        Create synthetic data where one position has extreme scores but low actual rate.
        Calibration should deflate those scores.
        """
        np.random.seed(123)
        n_events = 200
        scores = np.random.rand(n_events, 39)

        # Position 12 has very high scores but appears rarely in labels
        scores[:, 12] = np.random.rand(n_events) * 0.5 + 0.5  # [0.5, 1.0]

        # Create labels where position 12 appears only 5% of the time
        labels = np.zeros((n_events, 39), dtype=int)
        for i in range(n_events):
            winners = np.random.choice(39, 5, replace=False, p=None)
            labels[i, winners] = 1

        # Force position 12 to appear rarely
        labels[:, 12] = 0
        labels[:10, 12] = 1  # Only 10/200 = 5%

        cal = PositionIsotonicCalibrator()
        cal.fit(scores, labels)
        probs = cal.transform(scores)

        # After calibration, position 12's probabilities should be lower
        # (reflecting its true 5% rate, not the high raw scores)
        assert probs[:, 12].mean() < scores[:, 12].mean()


class TestHitsAtK:
    """Tests for _hits_at_k helper function."""

    def test_perfect_top20_prediction(self):
        """Test hits_at_k with perfect Top-20 predictions."""
        np.random.seed(42)
        n_events = 50

        # Create labels with exactly 5 ones per row
        y_true = np.zeros((n_events, 39), dtype=int)
        for i in range(n_events):
            y_true[i, np.random.choice(39, 5, replace=False)] = 1

        # Create perfect probabilities (1.0 for winners, 0.0 for losers)
        p = y_true.astype(float)

        hits = _hits_at_k(y_true, p, k=20)

        # With perfect probabilities, all 5 winners should be in Top-20
        assert hits.shape == (n_events,)
        npt.assert_array_equal(hits, 5)

    def test_random_prediction_hits(self):
        """Test hits_at_k with random probabilities."""
        np.random.seed(42)
        n_events = 100

        y_true = np.zeros((n_events, 39), dtype=int)
        for i in range(n_events):
            y_true[i, np.random.choice(39, 5, replace=False)] = 1

        # Random probabilities
        p = np.random.rand(n_events, 39)

        hits = _hits_at_k(y_true, p, k=20)

        # With random Top-20, expect ~2-3 hits on average (20/39 * 5 ≈ 2.56)
        assert hits.shape == (n_events,)
        assert 0 <= hits.min() <= 5
        assert 0 <= hits.max() <= 5
        assert 1.5 <= hits.mean() <= 3.5  # Reasonable range for random

    def test_zero_k_returns_zero_hits(self):
        """Test that k=0 returns zero hits."""
        y_true = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1] + [0] * 30])
        p = np.random.rand(1, 39)

        hits = _hits_at_k(y_true, p, k=0)
        assert hits[0] == 0


class TestRecallAt20:
    """Tests for recall_at_20 function."""

    def test_perfect_prediction_returns_100(self):
        """Test recall with perfect Top-20 predictions."""
        np.random.seed(42)
        n_events = 50

        y_true = np.zeros((n_events, 39), dtype=int)
        for i in range(n_events):
            y_true[i, np.random.choice(39, 5, replace=False)] = 1

        # Perfect probabilities
        p = y_true.astype(float)

        recall = recall_at_20(y_true, p)
        assert recall == 100.0

    def test_random_baseline_near_expected(self):
        """
        Test recall with random scores near 25.6% baseline.

        Random selection: 20 chosen from 39, 5 are winners
        Expected recall = (20/39) * 100 ≈ 51.3% or (5 * 20/39)/5 * 100 = 51.3%
        Actually for lottery: E[hits] = 5 * (20/39) ≈ 2.56
        Recall@20 = 2.56/5 * 100 ≈ 51.3%
        """
        np.random.seed(42)
        n_events = 500  # More events for stable average

        y_true = np.zeros((n_events, 39), dtype=int)
        for i in range(n_events):
            y_true[i, np.random.choice(39, 5, replace=False)] = 1

        # Random probabilities
        p = np.random.rand(n_events, 39)

        recall = recall_at_20(y_true, p)

        # With large sample, random recall should be near 51.3%
        assert (
            45.0 <= recall <= 57.0
        ), f"Random recall {recall:.2f}% outside expected range"

    def test_worst_case_prediction_returns_zero(self):
        """Test recall with worst-case predictions (all winners ranked last)."""
        y_true = np.array([[1, 1, 1, 1, 1] + [0] * 34])  # First 5 are winners

        # Give winners lowest scores (will be ranked last)
        p = np.zeros(39)
        p[:5] = 0.0  # Winners get 0.0
        p[5:] = 1.0  # Non-winners get 1.0
        p = p.reshape(1, 39)

        recall = recall_at_20(y_true, p)
        assert recall == 0.0

    def test_recall_returns_float(self):
        """Test that recall_at_20 returns a float."""
        y_true = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1] + [0] * 30])
        p = np.random.rand(1, 39)

        recall = recall_at_20(y_true, p)
        assert isinstance(recall, float)


class TestDistributionBreakdown:
    """Tests for distribution_breakdown function."""

    def test_perfect_predictions_all_01_wrong(self):
        """Test distribution with perfect predictions (all 0-1 wrong / 4-5 hits)."""
        np.random.seed(42)
        n_events = 100

        y_true = np.zeros((n_events, 39), dtype=int)
        for i in range(n_events):
            y_true[i, np.random.choice(39, 5, replace=False)] = 1

        # Perfect probabilities (all 5 winners in Top-20)
        p = y_true.astype(float)

        dist = distribution_breakdown(y_true, p, k=20)

        assert "0-1_wrong" in dist
        assert "2-3_wrong" in dist
        assert "4-5_wrong" in dist

        # All events should have 0-1 wrong (5 hits = 0 wrong)
        assert dist["0-1_wrong"] == 100.0
        assert dist["2-3_wrong"] == 0.0
        assert dist["4-5_wrong"] == 0.0

    def test_worst_predictions_all_45_wrong(self):
        """Test distribution with worst predictions (all 4-5 wrong / 0-1 hits)."""
        n_events = 100

        y_true = np.zeros((n_events, 39), dtype=int)
        for i in range(n_events):
            y_true[i, :5] = 1  # First 5 positions are always winners

        # Give winners lowest scores so they're never in Top-20
        p = np.zeros((n_events, 39))
        p[:, :5] = 0.0  # Winners get 0.0
        p[:, 5:] = 1.0  # Non-winners get 1.0

        dist = distribution_breakdown(y_true, p, k=20)

        # All events should have 4-5 wrong (0-1 hits)
        assert dist["0-1_wrong"] == 0.0
        assert dist["2-3_wrong"] == 0.0
        assert dist["4-5_wrong"] == 100.0

    def test_distribution_sums_to_approximately_100(self):
        """Test that distribution percentages sum to ~100%."""
        np.random.seed(42)
        n_events = 200

        y_true = np.zeros((n_events, 39), dtype=int)
        for i in range(n_events):
            y_true[i, np.random.choice(39, 5, replace=False)] = 1

        p = np.random.rand(n_events, 39)

        dist = distribution_breakdown(y_true, p, k=20)

        total = dist["0-1_wrong"] + dist["2-3_wrong"] + dist["4-5_wrong"]
        assert 99.0 <= total <= 101.0, f"Distribution total {total:.1f}% not near 100%"

    def test_mixed_distribution(self):
        """Test distribution with known mixed outcomes."""
        # Manually create a known distribution
        # 20 events with 5 hits, 30 events with 3 hits, 50 events with 1 hit
        n_events = 100
        y_true = np.zeros((n_events, 39), dtype=int)
        p = np.zeros((n_events, 39))

        # Events 0-19: 5 hits (0 wrong) → 0-1 wrong category
        for i in range(20):
            winners = np.random.choice(39, 5, replace=False)
            y_true[i, winners] = 1
            p[i, winners] = 1.0  # Perfect prediction

        # Events 20-49: 3 hits (2 wrong) → 2-3 wrong category
        for i in range(20, 50):
            winners = np.random.choice(39, 5, replace=False)
            y_true[i, winners] = 1
            # Give 3 winners high scores, 2 low scores
            p[i, winners[:3]] = 1.0
            p[i, winners[3:]] = 0.0
            p[i, ~np.isin(np.arange(39), winners)] = 0.5

        # Events 50-99: 1 hit (4 wrong) → 4-5 wrong category
        for i in range(50, 100):
            winners = np.random.choice(39, 5, replace=False)
            y_true[i, winners] = 1
            # Give 1 winner high score, 4 low scores
            p[i, winners[0]] = 1.0
            p[i, winners[1:]] = 0.0
            p[i, ~np.isin(np.arange(39), winners)] = 0.5

        dist = distribution_breakdown(y_true, p, k=20)

        # Expected: 20% in 0-1 wrong, 30% in 2-3 wrong, 50% in 4-5 wrong
        assert 18.0 <= dist["0-1_wrong"] <= 22.0
        assert 28.0 <= dist["2-3_wrong"] <= 32.0
        assert 48.0 <= dist["4-5_wrong"] <= 52.0

    def test_returns_dict_with_correct_keys(self):
        """Test that distribution_breakdown returns dict with expected keys."""
        y_true = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1] + [0] * 30])
        p = np.random.rand(1, 39)

        dist = distribution_breakdown(y_true, p, k=20)

        assert isinstance(dist, dict)
        assert set(dist.keys()) == {"0-1_wrong", "2-3_wrong", "4-5_wrong"}
        assert all(isinstance(v, float) for v in dist.values())


class TestCalibrationPackConfig:
    """Tests for CalibrationPackConfig dataclass."""

    def test_default_normalize_is_true(self):
        """Test that default normalize is True."""
        config = CalibrationPackConfig()
        assert config.normalize is True

    def test_custom_normalize_false(self):
        """Test that custom normalize=False works."""
        config = CalibrationPackConfig(normalize=False)
        assert config.normalize is False


class TestCLIInterface:
    """Tests for CLI interface (fit and apply modes)."""

    @pytest.fixture
    def temp_data_files(self):
        """Create temporary parquet files for CLI testing."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create synthetic scores and labels
            np.random.seed(42)
            n_events = 50
            scores_data = {
                "event_id": list(range(n_events)),
                **{f"p{i}": np.random.rand(n_events) for i in range(1, 40)},
            }
            labels_data = {
                "event_id": list(range(n_events)),
            }
            # Create labels with exactly 5 ones per row
            for i in range(1, 40):
                labels_data[f"y{i}"] = [0] * n_events
            for i in range(n_events):
                winners = np.random.choice(range(1, 40), 5, replace=False)
                for w in winners:
                    labels_data[f"y{w}"][i] = 1

            scores_df = pd.DataFrame(scores_data)
            labels_df = pd.DataFrame(labels_data)

            scores_path = tmpdir / "scores.parquet"
            labels_path = tmpdir / "labels.parquet"

            scores_df.to_parquet(scores_path, index=False)
            labels_df.to_parquet(labels_path, index=False)

            yield {
                "tmpdir": tmpdir,
                "scores": scores_path,
                "labels": labels_path,
            }

    def test_cli_fit_mode_creates_model_and_output(self, temp_data_files):
        """Test CLI fit mode creates model file and calibrated output."""
        import subprocess

        tmpdir = temp_data_files["tmpdir"]
        model_path = tmpdir / "calibrator.joblib"
        output_path = tmpdir / "calibrated.parquet"

        # Run CLI fit command
        cmd = [
            "python",
            "-m",
            "src.modeling.calibration.position_isotonic",
            "fit",
            "--scores",
            str(temp_data_files["scores"]),
            "--labels",
            str(temp_data_files["labels"]),
            "--save-model",
            str(model_path),
            "--out-calibrated",
            str(output_path),
            "--normalize",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check command succeeded
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Check model file was created
        assert model_path.exists()

        # Check calibrated output was created
        assert output_path.exists()

        # Verify output can be loaded
        import pandas as pd

        df = pd.read_parquet(output_path)
        assert "event_id" in df.columns
        assert all(f"p{i}" in df.columns for i in range(1, 40))

    def test_cli_fit_mode_with_topk_output(self, temp_data_files):
        """Test CLI fit mode with optional Top-20 indices output."""
        import subprocess

        tmpdir = temp_data_files["tmpdir"]
        model_path = tmpdir / "calibrator.joblib"
        output_path = tmpdir / "calibrated.parquet"
        topk_path = tmpdir / "topk.parquet"

        cmd = [
            "python",
            "-m",
            "src.modeling.calibration.position_isotonic",
            "fit",
            "--scores",
            str(temp_data_files["scores"]),
            "--labels",
            str(temp_data_files["labels"]),
            "--save-model",
            str(model_path),
            "--out-calibrated",
            str(output_path),
            "--out-topk",
            str(topk_path),
            "--normalize",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0
        assert topk_path.exists()

        # Verify Top-20 output
        import pandas as pd

        topk_df = pd.read_parquet(topk_path)
        assert "event_id" in topk_df.columns
        assert all(f"top20_{i}" in topk_df.columns for i in range(1, 21))

        # Verify indices are in valid range [1, 39]
        for col in [f"top20_{i}" for i in range(1, 21)]:
            assert topk_df[col].min() >= 1
            assert topk_df[col].max() <= 39

    def test_cli_apply_mode_uses_saved_model(self, temp_data_files):
        """Test CLI apply mode loads and applies saved calibrator."""
        import subprocess
        import pandas as pd

        tmpdir = temp_data_files["tmpdir"]
        model_path = tmpdir / "calibrator.joblib"
        output_path = tmpdir / "calibrated.parquet"

        # First, fit and save a model
        cmd_fit = [
            "python",
            "-m",
            "src.modeling.calibration.position_isotonic",
            "fit",
            "--scores",
            str(temp_data_files["scores"]),
            "--labels",
            str(temp_data_files["labels"]),
            "--save-model",
            str(model_path),
            "--out-calibrated",
            str(output_path),
        ]
        subprocess.run(cmd_fit, check=True, capture_output=True)

        # Now apply the saved model to new scores
        apply_output = tmpdir / "applied.parquet"
        cmd_apply = [
            "python",
            "-m",
            "src.modeling.calibration.position_isotonic",
            "apply",
            "--scores",
            str(temp_data_files["scores"]),
            "--load-model",
            str(model_path),
            "--out-calibrated",
            str(apply_output),
        ]

        result = subprocess.run(cmd_apply, capture_output=True, text=True)

        assert result.returncode == 0
        assert apply_output.exists()

        # Verify apply output matches fit output (same input scores)
        fit_df = pd.read_parquet(output_path)
        apply_df = pd.read_parquet(apply_output)

        pd.testing.assert_frame_equal(fit_df, apply_df)
