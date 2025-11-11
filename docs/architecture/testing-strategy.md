# Testing Strategy - C5 Quantum Imputer

**Version**: 1.0
**Last Updated**: 2025-11-11

## Testing Philosophy

**Core Principle**: Tests alongside each story, not batched at epic completion.

**Coverage Target**: ≥90% for all new modules (Epic 14-17)

**Validation Approach**: Strict rolling/blocked cross-validation (no temporal leakage)

## Test Pyramid

### Unit Tests (70% of tests)
- Test individual functions and methods in isolation
- Fast execution (<1s per test)
- Mock external dependencies
- **Location**: `tests/modeling/{module}/test_{filename}.py`

### Integration Tests (25% of tests)
- Test module interactions and data pipelines
- Use synthetic data or small real datasets
- Validate end-to-end workflows
- **Location**: `tests/integration/test_{epic}_pipeline.py`

### End-to-End Tests (5% of tests)
- Test complete pipeline on 1000-event holdout
- Validate performance metrics (Recall@20, middle-zone distribution)
- **Location**: `tests/e2e/test_epic{N}_e2e.py`

## Testing Framework

### Tools
- **pytest**: Test runner and framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities (if needed)

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run specific module tests
pytest tests/modeling/calibration/test_position_isotonic.py -v

# Run integration tests only
pytest tests/integration/ -v
```

## Test File Structure

### Naming Convention
- Test files: `test_{module_name}.py`
- Test classes: `Test{ClassName}`
- Test functions: `test_{function_name}_{scenario}`

### Example Structure
```python
"""Tests for position isotonic calibration module."""
import numpy as np
import pytest
from src.modeling.calibration.position_isotonic import (
    PositionIsotonicCalibrator,
    recall_at_20,
    distribution_breakdown,
)


class TestPositionIsotonicCalibrator:
    """Tests for PositionIsotonicCalibrator class."""

    def test_fit_creates_39_models(self):
        """Test that fit creates one isotonic regressor per position."""
        pass

    def test_transform_output_shape(self):
        """Test that transform returns correct shape."""
        pass

    def test_transform_probabilities_sum_to_one(self):
        """Test that probabilities are normalized per event."""
        pass


class TestRecallAt20:
    """Tests for recall_at_20 function."""

    def test_perfect_prediction_returns_100(self):
        """Test recall with perfect Top-20 predictions."""
        pass

    def test_random_baseline_near_25_percent(self):
        """Test recall with random scores near 25.6% baseline."""
        pass
```

## Test Data Strategy

### Synthetic Data Generation
- Create deterministic synthetic datasets for unit tests
- Use `np.random.seed()` for reproducibility
- Example:
```python
@pytest.fixture
def synthetic_scores():
    """Generate synthetic score matrix (n_events, 39)."""
    np.random.seed(42)
    return np.random.rand(100, 39)

@pytest.fixture
def synthetic_labels():
    """Generate synthetic binary labels (n_events, 39) with exactly 5 ones per row."""
    np.random.seed(42)
    labels = np.zeros((100, 39), dtype=int)
    for i in range(100):
        labels[i, np.random.choice(39, 5, replace=False)] = 1
    return labels
```

### Real Data for Integration Tests
- Use small slices of `data/raw/c5_Matrix.csv`
- Limit to 100-200 events for speed
- Store test fixtures in `tests/fixtures/` if needed

### Holdout Data for E2E Tests
- Use the 1000-event holdout from `production/comprehensive_1000event_holdout_diagnostic.py`
- Only run E2E tests on story completion (not during development)

## Test Coverage Requirements

### Per-Story Requirements
- **Unit tests**: ≥90% coverage for story's new code
- **Integration test**: At least 1 test validating module interaction
- **Acceptance criteria**: All AC must have corresponding test validation

### Coverage Exemptions
- CLI `if __name__ == "__main__"` blocks (covered by manual testing)
- Trivial getters/setters (if any)
- Plotting/visualization code (Epic 17 only)

## Test Assertions

### NumPy Array Assertions
```python
import numpy.testing as npt

# Exact equality (integers, booleans)
npt.assert_array_equal(actual, expected)

# Floating-point equality (with tolerance)
npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

# Shape assertions
assert actual.shape == (100, 39), f"Expected shape (100, 39), got {actual.shape}"
```

### Statistical Assertions
```python
# Recall within expected range
assert 50.0 <= recall <= 55.0, f"Recall {recall:.2f}% outside expected range"

# Distribution sums to ~100%
total = dist["0-1_wrong"] + dist["2-3_wrong"] + dist["4-5_wrong"]
assert 99.0 <= total <= 101.0, f"Distribution total {total:.1f}% not near 100%"
```

### Exception Testing
```python
with pytest.raises(AssertionError, match="Calibrator not fitted"):
    calibrator.transform(scores)  # Should fail if not fitted
```

## Mocking and Fixtures

### When to Mock
- External file I/O (for unit tests)
- Expensive computations (use precomputed results)
- Random number generation (for determinism)

### Pytest Fixtures
```python
@pytest.fixture
def calibrator():
    """Provide a fitted calibrator instance."""
    cal = PositionIsotonicCalibrator()
    # Fit on synthetic data
    scores = np.random.rand(100, 39)
    labels = np.random.randint(0, 2, (100, 39))
    cal.fit(scores, labels)
    return cal
```

## Integration Test Patterns

### Pipeline Testing
```python
def test_calibration_to_extremizer_pipeline():
    """Test that calibrator output can be consumed by extremizer."""
    # Step 1: Calibrate
    calibrator = PositionIsotonicCalibrator()
    calibrator.fit(train_scores, train_labels)
    cal_probs = calibrator.transform(test_scores)

    # Step 2: Apply extremizer
    extremizer = Extremizer()
    extremizer.fit(cal_probs, test_labels)
    final_probs = extremizer.transform(cal_probs)

    # Validate output shape and properties
    assert final_probs.shape == test_scores.shape
    assert np.allclose(final_probs.sum(axis=1), 1.0, atol=1e-6)
```

### Metric Validation
```python
def test_middle_zone_reduction():
    """Validate that extremizer reduces middle-zone percentage."""
    # Baseline distribution
    baseline_dist = distribution_breakdown(labels, baseline_probs)

    # Extremizer distribution
    extremizer_dist = distribution_breakdown(labels, extremizer_probs)

    # Assert reduction
    reduction = baseline_dist["2-3_wrong"] - extremizer_dist["2-3_wrong"]
    assert reduction >= 8.0, f"Middle-zone reduction {reduction:.1f}pp below target (8pp)"
```

## Test Organization by Epic

### Epic 14: Extremizer Meta-Layer
```
tests/modeling/
├── calibration/
│   └── test_position_isotonic.py       (Story 14.1)
├── meta/
│   ├── test_diagnostics.py              (Story 14.2)
│   ├── test_risk_classifier.py          (Story 14.3)
│   ├── test_temperature_shaping.py      (Story 14.4)
│   └── test_extremizer.py               (Story 14.5)
└── integration/
    └── test_epic14_pipeline.py          (Story 14.5)
```

### Epic 15: Ising Re-Ranker
```
tests/modeling/ensembles/
├── test_ising_stats.py                  (Story 15.1)
├── test_ising_parameters.py             (Story 15.2)
├── test_synergy_augmentation.py         (Story 15.3)
├── test_ising_reranker.py               (Story 15.4)
└── integration/
    └── test_epic15_pipeline.py          (Story 15.5)
```

## Continuous Integration (Future)

### GitHub Actions Workflow
- Run tests on every push
- Generate coverage reports
- Fail PR if coverage <90% or tests fail
- **File**: `.github/workflows/test.yml` (to be created in Epic 17)

## Performance Testing

### Timing Assertions
```python
import time

def test_calibration_performance():
    """Test that calibration completes within 5s for 1000 events."""
    scores = np.random.rand(1000, 39)
    labels = np.random.randint(0, 2, (1000, 39))

    cal = PositionIsotonicCalibrator()

    start = time.time()
    cal.fit(scores, labels)
    cal.transform(scores)
    elapsed = time.time() - start

    assert elapsed < 5.0, f"Calibration took {elapsed:.2f}s, exceeds 5s limit"
```

## Test Documentation

### Docstrings for Tests
- Brief description of what is being tested
- Expected behavior
- Example:
```python
def test_transform_output_normalized():
    """
    Test that transform output is normalized.

    Each event's probabilities should sum to 1.0 (within numerical tolerance).
    """
    pass
```

## Definition of Done (Testing)

For a story to be marked complete:
- [ ] All unit tests written and passing
- [ ] Coverage ≥90% for story's code
- [ ] Integration test(s) passing
- [ ] All acceptance criteria have test validation
- [ ] No skipped tests (unless documented with reason)
- [ ] Tests run in <30s (unit + integration)
