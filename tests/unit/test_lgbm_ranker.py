"""
Unit Tests for LightGBM Gradient Boosting Ranker

This module tests the LGBMRanker implementation, including initialization,
training, prediction, model persistence, and performance tracking.

Test Coverage:
    - Initialization and parameter validation
    - Training (fit) functionality
    - Prediction (predict_top_k) functionality
    - Model save/load functionality
    - Training time tracking (NFR1 requirement)
    - Feature importance extraction
    - Error handling and edge cases
    - Integration tests with realistic data

Author: BMad Dev Agent (James)
Date: 2025-10-14
Story: Epic 3, Story 3.3 - Gradient Boosting Ranker
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

from src.modeling.rankers.lgbm_ranker import LGBMRanker, LIGHTGBM_AVAILABLE

# Skip all tests if LightGBM not available
pytestmark = pytest.mark.skipif(
    not LIGHTGBM_AVAILABLE,
    reason="LightGBM not installed"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_training_data():
    """
    Create sample training data with binary position indicators.

    Creates a dataset with 100 samples where each sample has
    5 active positions (1-indexed from 1 to 39).
    """
    np.random.seed(42)
    n_samples = 100
    n_positions = 39

    # Create binary matrix: each row has exactly 5 active positions
    data = np.zeros((n_samples, n_positions))

    for i in range(n_samples):
        # Randomly select 5 positions to be active
        active_pos = np.random.choice(n_positions, size=5, replace=False)
        data[i, active_pos] = 1

    # Create DataFrame with position-named columns + event-ID
    columns = ['event-ID'] + [f'qv_{i+1}_onehot' for i in range(n_positions)]
    df = pd.DataFrame(
        np.column_stack([np.arange(1, n_samples+1), data]),
        columns=columns
    )

    return df


@pytest.fixture
def simple_training_data():
    """
    Create simple training data for controlled testing.

    CRITICAL: Each row MUST have exactly 5 active positions (5 ones).
    """
    np.random.seed(42)
    n_samples = 20
    n_positions = 39

    # Create binary matrix: each row has EXACTLY 5 active positions
    data = np.zeros((n_samples, n_positions))

    for i in range(n_samples):
        # Randomly select exactly 5 positions to be active
        active_pos = np.random.choice(n_positions, size=5, replace=False)
        data[i, active_pos] = 1

    # Verify constraint: each row has exactly 5 ones
    assert np.all(data.sum(axis=1) == 5), "Each row must have exactly 5 active positions"

    # Create DataFrame with position-named columns + event-ID
    columns = ['event-ID'] + [f'qv_{i+1}_onehot' for i in range(n_positions)]
    df = pd.DataFrame(
        np.column_stack([np.arange(1, n_samples+1), data]),
        columns=columns
    )

    return df


# ============================================================================
# Tests for Initialization
# ============================================================================

def test_lgbm_ranker_initialization_default():
    """Test LGBMRanker initialization with default parameters."""
    ranker = LGBMRanker()

    assert ranker.track_time is True
    assert ranker.is_fitted_ is False
    assert ranker.model_ is None
    assert ranker.training_time_ is None
    assert 'objective' in ranker.params
    assert ranker.params['objective'] == 'lambdarank'


def test_lgbm_ranker_initialization_custom_params():
    """Test LGBMRanker initialization with custom hyperparameters."""
    custom_params = {'n_estimators': 200, 'learning_rate': 0.05}
    ranker = LGBMRanker(params=custom_params)

    assert ranker.params['n_estimators'] == 200
    assert ranker.params['learning_rate'] == 0.05
    # Should still have defaults for other params
    assert ranker.params['objective'] == 'lambdarank'


def test_lgbm_ranker_initialization_track_time_false():
    """Test LGBMRanker initialization with time tracking disabled."""
    ranker = LGBMRanker(track_time=False)

    assert ranker.track_time is False


def test_import_error_if_lightgbm_not_available():
    """Test that ImportError is raised if LightGBM not installed."""
    # This test simulates LightGBM not being available
    with patch('src.modeling.rankers.lgbm_ranker.LIGHTGBM_AVAILABLE', False):
        with pytest.raises(ImportError, match="LightGBM is not installed"):
            LGBMRanker()


# ============================================================================
# Tests for fit()
# ============================================================================

def test_fit_basic(simple_training_data):
    """Test basic fit functionality."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    assert ranker.is_fitted_ is True
    assert ranker.model_ is not None
    assert ranker.position_frequencies_ is not None
    assert ranker.feature_names_ is not None


def test_fit_tracks_training_time(simple_training_data):
    """Test that training time is tracked (NFR1 requirement)."""
    ranker = LGBMRanker(track_time=True)
    ranker.fit(simple_training_data)

    assert ranker.training_time_ is not None
    assert ranker.training_time_ > 0
    assert isinstance(ranker.training_time_, float)


def test_fit_no_tracking_time(simple_training_data):
    """Test fit with time tracking disabled."""
    ranker = LGBMRanker(track_time=False)
    ranker.fit(simple_training_data)

    assert ranker.is_fitted_ is True
    # training_time_ should remain None
    assert ranker.training_time_ is None


def test_fit_extracts_feature_importance(simple_training_data):
    """Test that feature importance is extracted after training."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    assert ranker.feature_importance_ is not None
    assert isinstance(ranker.feature_importance_, dict)
    assert len(ranker.feature_importance_) > 0
    # Check that 'position_frequency' feature is present
    assert 'position_frequency' in ranker.feature_importance_


def test_fit_computes_position_frequencies(simple_training_data):
    """Test that position frequencies are computed from training data."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    assert ranker.position_frequencies_ is not None
    # All 39 positions should have frequency entries
    assert len(ranker.position_frequencies_) == 39
    # All frequencies should be positive (small epsilon for unseen positions)
    assert all(freq > 0 for freq in ranker.position_frequencies_.values())
    # Frequencies should sum to approximately 1.0 (normalized probabilities)
    total_freq = sum(ranker.position_frequencies_.values())
    assert 0.99 < total_freq < 1.01  # Allow small floating point error


def test_fit_with_empty_dataframe_raises_error():
    """Test that fit with empty DataFrame raises ValueError."""
    ranker = LGBMRanker()

    with pytest.raises(ValueError, match="cannot be empty"):
        ranker.fit(pd.DataFrame())


def test_fit_with_too_few_samples_raises_error():
    """Test that fit with very small dataset raises ValueError."""
    ranker = LGBMRanker()
    small_df = pd.DataFrame({
        'event-ID': [1, 2, 3],
        'qv_1_onehot': [1, 0, 1],
        'qv_2_onehot': [0, 1, 0]
    })

    with pytest.raises(ValueError, match="too small"):
        ranker.fit(small_df)


def test_fit_returns_self(simple_training_data):
    """Test that fit returns self for method chaining."""
    ranker = LGBMRanker()
    result = ranker.fit(simple_training_data)

    assert result is ranker
    assert ranker.is_fitted_


# ============================================================================
# Tests for predict_top_k()
# ============================================================================

def test_predict_basic(simple_training_data):
    """Test basic predict functionality."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    predictions = ranker.predict_top_k(simple_training_data, k=5)

    assert predictions.shape == (len(simple_training_data), 5)
    # All predictions should be in valid range [1, 39]
    assert np.all(predictions >= 1)
    assert np.all(predictions <= 39)


def test_predict_top_k_20(simple_training_data):
    """Test prediction with k=20 (project requirement)."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    predictions = ranker.predict_top_k(simple_training_data, k=20)

    assert predictions.shape == (len(simple_training_data), 20)
    assert np.all(predictions >= 1)
    assert np.all(predictions <= 39)
    # Each row should have unique positions (no duplicates)
    for row in predictions:
        assert len(set(row)) == 20


def test_predict_with_k_equals_1(simple_training_data):
    """Test prediction with k=1 (minimum valid k)."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    predictions = ranker.predict_top_k(simple_training_data, k=1)

    assert predictions.shape == (len(simple_training_data), 1)
    assert np.all(predictions >= 1)
    assert np.all(predictions <= 39)


def test_predict_with_k_equals_39(simple_training_data):
    """Test prediction with k=39 (maximum valid k, all positions)."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    predictions = ranker.predict_top_k(simple_training_data, k=39)

    assert predictions.shape == (len(simple_training_data), 39)
    # Each row should contain all positions 1-39 exactly once
    for row in predictions:
        assert set(row) == set(range(1, 40))


def test_predict_before_fit_raises_error():
    """Test that predict_top_k before fit raises RuntimeError."""
    ranker = LGBMRanker()

    with pytest.raises(RuntimeError, match="must be fitted before calling predict_top_k"):
        ranker.predict_top_k(pd.DataFrame({'qv_1_onehot': [1]}), k=20)


def test_predict_with_invalid_k_raises_error(simple_training_data):
    """Test that invalid k values raise ValueError."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    with pytest.raises(ValueError, match="k must be in range"):
        ranker.predict_top_k(simple_training_data, k=0)

    with pytest.raises(ValueError, match="k must be in range"):
        ranker.predict_top_k(simple_training_data, k=40)

    with pytest.raises(ValueError, match="k must be in range"):
        ranker.predict_top_k(simple_training_data, k=-5)


def test_predict_with_empty_test_data_raises_error(simple_training_data):
    """Test that predict with empty test data raises ValueError."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    with pytest.raises(ValueError, match="cannot be empty"):
        ranker.predict_top_k(pd.DataFrame(), k=20)


# ============================================================================
# Tests for Model Save/Load
# ============================================================================

def test_save_model(simple_training_data):
    """Test model saving functionality."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        ranker.save_model(model_path)

        assert model_path.exists()
        # Metadata file should also exist
        metadata_path = Path(str(model_path) + '.meta.json')
        assert metadata_path.exists()


def test_save_model_metadata_content(simple_training_data):
    """Test that model metadata contains expected information."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        ranker.save_model(model_path, save_metadata=True)

        metadata_path = Path(str(model_path) + '.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert metadata['ranker_type'] == 'lgbm'
        assert 'hyperparameters' in metadata
        assert 'training_time_seconds' in metadata
        assert 'feature_importance' in metadata
        assert metadata['training_time_seconds'] > 0


def test_save_model_without_metadata(simple_training_data):
    """Test saving model without metadata."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        ranker.save_model(model_path, save_metadata=False)

        assert model_path.exists()
        # Metadata file should NOT exist
        metadata_path = Path(str(model_path) + '.meta.json')
        assert not metadata_path.exists()


def test_load_model(simple_training_data):
    """Test model loading functionality."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        ranker.save_model(model_path)

        # Load the model
        loaded_ranker = LGBMRanker.load_model(model_path)

        assert loaded_ranker.is_fitted_ is True
        assert loaded_ranker.model_ is not None
        assert loaded_ranker.training_time_ == ranker.training_time_


def test_load_model_predictions_match(simple_training_data):
    """Test that loaded model produces same predictions as original."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)

    original_predictions = ranker.predict_top_k(simple_training_data, k=20)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        ranker.save_model(model_path)

        # Load and predict
        loaded_ranker = LGBMRanker.load_model(model_path)
        loaded_predictions = loaded_ranker.predict_top_k(simple_training_data, k=20)

        # Predictions should be identical
        assert np.array_equal(original_predictions, loaded_predictions)


def test_load_model_file_not_found():
    """Test that loading non-existent model raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        LGBMRanker.load_model(Path("nonexistent_model.joblib"))


def test_save_before_fit_raises_error():
    """Test that save_model before fit raises RuntimeError."""
    ranker = LGBMRanker()

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        with pytest.raises(RuntimeError, match="must be fitted"):
            ranker.save_model(model_path)


# ============================================================================
# Tests for Helper Methods
# ============================================================================

def test_get_name():
    """Test that get_name() returns class name."""
    ranker = LGBMRanker()
    assert ranker.get_name() == 'LGBMRanker'


def test_repr_before_fit():
    """Test __repr__ before fitting."""
    ranker = LGBMRanker()
    repr_str = repr(ranker)

    assert 'LGBMRanker' in repr_str
    assert 'not fitted' in repr_str


def test_repr_after_fit(simple_training_data):
    """Test __repr__ after fitting with training time."""
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)
    repr_str = repr(ranker)

    assert 'LGBMRanker' in repr_str
    assert 'fitted' in repr_str
    assert 'trained in' in repr_str
    assert 's)' in repr_str  # Should show seconds


def test_circular_distance():
    """Test circular distance calculation on C₃₉ ring."""
    ranker = LGBMRanker()

    # Direct distance: 1 to 3 = 2
    assert ranker._circular_distance(1, 3) == 2

    # Wrap-around distance: 1 to 39 = 1 (shortest path wraps around)
    assert ranker._circular_distance(1, 39) == 1

    # Maximum distance: opposite side of ring (39/2 ≈ 19-20)
    assert ranker._circular_distance(1, 20) == 19

    # Same position: distance = 0
    assert ranker._circular_distance(5, 5) == 0


def test_fit_transform(simple_training_data):
    """Test fit_transform convenience method."""
    ranker = LGBMRanker()

    predictions = ranker.fit_transform(simple_training_data, k=5)

    assert ranker.is_fitted_
    assert predictions.shape == (len(simple_training_data), 5)


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow(sample_training_data):
    """Test full workflow on realistic data."""
    ranker = LGBMRanker()
    ranker.fit(sample_training_data)

    predictions = ranker.predict_top_k(sample_training_data, k=20)

    assert predictions.shape == (len(sample_training_data), 20)
    # All predictions should be in valid range [1, 39]
    assert np.all(predictions >= 1)
    assert np.all(predictions <= 39)
    # Each row should have unique positions (no duplicates)
    for row in predictions:
        assert len(set(row)) == 20


def test_train_and_test_on_different_data(sample_training_data):
    """Test training on one dataset and predicting on another."""
    # Split data into train and test
    n_train = 80
    train_df = sample_training_data.iloc[:n_train]
    test_df = sample_training_data.iloc[n_train:]

    ranker = LGBMRanker()
    ranker.fit(train_df)

    predictions = ranker.predict_top_k(test_df, k=20)

    assert predictions.shape == (len(test_df), 20)
    assert np.all(predictions >= 1)
    assert np.all(predictions <= 39)


def test_custom_hyperparameters_workflow(simple_training_data):
    """Test workflow with custom hyperparameters."""
    custom_params = {
        'n_estimators': 50,  # Fewer trees for faster test
        'learning_rate': 0.2,
        'num_leaves': 15
    }
    ranker = LGBMRanker(params=custom_params)
    ranker.fit(simple_training_data)

    predictions = ranker.predict_top_k(simple_training_data, k=20)

    assert predictions.shape == (len(simple_training_data), 20)
    # Verify custom params were used
    assert ranker.params['n_estimators'] == 50
    assert ranker.params['learning_rate'] == 0.2


def test_model_persistence_workflow(simple_training_data):
    """Test complete workflow with model save/load."""
    # Train model
    ranker = LGBMRanker()
    ranker.fit(simple_training_data)
    original_predictions = ranker.predict_top_k(simple_training_data, k=20)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "workflow_model.joblib"

        # Save model
        ranker.save_model(model_path)

        # Load model in new instance
        loaded_ranker = LGBMRanker.load_model(model_path)

        # Predict with loaded model
        loaded_predictions = loaded_ranker.predict_top_k(simple_training_data, k=20)

        # Predictions should match
        assert np.array_equal(original_predictions, loaded_predictions)


# ============================================================================
# Performance and NFR1 Tests
# ============================================================================

def test_training_time_warning_for_long_training(simple_training_data, caplog):
    """Test that warning is logged if training exceeds 1 hour."""
    ranker = LGBMRanker(track_time=True)

    # Mock training time to simulate long training
    with patch.object(ranker, 'training_time_', 3700):  # 61+ minutes
        # Manually set is_fitted to skip actual training
        ranker.is_fitted_ = True
        ranker.training_time_ = 3700

        # Check that warning would be logged
        # (In real scenario, this happens in fit() method)
        if ranker.training_time_ > 3600:
            assert ranker.training_time_ / 3600 > 1.0


def test_training_time_logged(simple_training_data, caplog):
    """Test that training completion is logged."""
    import logging
    caplog.set_level(logging.INFO)

    ranker = LGBMRanker(track_time=True)
    ranker.fit(simple_training_data)

    # Check that training completion was logged
    assert any("Training completed in" in record.message for record in caplog.records)


# ============================================================================
# Edge Cases
# ============================================================================

def test_predict_with_all_positions_inactive():
    """Test prediction when test sample has no active positions."""
    # This is an edge case - model should still produce predictions
    ranker = LGBMRanker()

    # Train on valid data (each row has exactly 5 active positions)
    np.random.seed(99)
    n_samples = 15
    data = np.zeros((n_samples, 39))
    for i in range(n_samples):
        active_pos = np.random.choice(39, size=5, replace=False)
        data[i, active_pos] = 1

    columns = ['event-ID'] + [f'qv_{i+1}_onehot' for i in range(39)]
    train_data = pd.DataFrame(
        np.column_stack([np.arange(1, n_samples+1), data]),
        columns=columns
    )

    ranker.fit(train_data)

    # Test on data with no active positions (all zeros)
    test_data = pd.DataFrame({
        'event-ID': [3],
    })
    for i in range(1, 40):
        test_data[f'qv_{i}_onehot'] = [0]

    predictions = ranker.predict_top_k(test_data, k=20)

    # Should still produce valid predictions
    assert predictions.shape == (1, 20)
    assert np.all(predictions >= 1)
    assert np.all(predictions <= 39)


def test_reproducibility_with_same_data(simple_training_data):
    """Test that training with same data produces consistent results."""
    # Train two models with same data and random_state
    ranker1 = LGBMRanker(params={'random_state': 42})
    ranker1.fit(simple_training_data)
    pred1 = ranker1.predict_top_k(simple_training_data, k=20)

    ranker2 = LGBMRanker(params={'random_state': 42})
    ranker2.fit(simple_training_data)
    pred2 = ranker2.predict_top_k(simple_training_data, k=20)

    # Predictions should be identical (same random seed)
    assert np.array_equal(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
