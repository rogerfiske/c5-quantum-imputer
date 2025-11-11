"""
Unit Tests for Evaluation Module

Tests metrics computation, holdout test runner, and report generation.

Author: BMad Dev Agent (James)
Date: 2025-10-14
Epic: Epic 4 - Evaluation and Reporting
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.holdout_test import HoldoutTestRunner
from src.evaluation.metrics import (
    WrongPredictionsMetric,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_top_k_accuracy,
)
from src.evaluation.report_generator import ReportGenerator, compare_rankers


@pytest.fixture
def sample_holdout_data():
    """Create sample holdout data for testing."""
    np.random.seed(42)

    n_events = 50
    data = {
        'event_id': list(range(1, n_events + 1)),
        'q_1': [1, 5, 10, 15, 20] * 10,  # Cycle through 5 positions
        'q_2': [2, 6, 11, 16, 21] * 10,
        'q_3': [3, 7, 12, 17, 22] * 10,
        'q_4': [4, 8, 13, 18, 23] * 10,
        'q_5': [5, 9, 14, 19, 24] * 10,
        'feature_1': np.random.randn(n_events),
        'feature_2': np.random.randn(n_events),
    }

    return pd.DataFrame(data)


@pytest.fixture
def mock_ranker():
    """Create a mock ranker for testing."""
    ranker = MagicMock()
    ranker.__class__.__name__ = 'MockRanker'

    def mock_predict_top_k(df, k=20):
        """Mock prediction that returns first k positions."""
        n_events = len(df)
        # Return positions 1-k for each event
        return [[i for i in range(1, k+1)] for _ in range(n_events)]

    ranker.predict_top_k = mock_predict_top_k
    ranker.params = {'test_param': 'value'}

    return ranker


class TestWrongPredictionsMetric:
    """Test WrongPredictionsMetric class."""

    def test_initialization(self):
        """Test metric initialization with default parameters."""
        metric = WrongPredictionsMetric()

        assert metric.k == 20
        assert metric.position_columns == ['q_1', 'q_2', 'q_3', 'q_4', 'q_5']

    def test_initialization_custom_params(self):
        """Test metric initialization with custom parameters."""
        metric = WrongPredictionsMetric(k=10, position_columns=['pos1', 'pos2', 'pos3'])

        assert metric.k == 10
        assert metric.position_columns == ['pos1', 'pos2', 'pos3']

    def test_compute_single_event_all_correct(self):
        """Test computing wrong count when all positions are in predictions."""
        metric = WrongPredictionsMetric(k=20)

        actual = [1, 5, 10, 15, 20]
        predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        wrong_count = metric.compute_single_event(actual, predicted)
        assert wrong_count == 0  # All 5 positions found in top-20

    def test_compute_single_event_all_wrong(self):
        """Test computing wrong count when no positions are in predictions."""
        metric = WrongPredictionsMetric(k=20)

        actual = [21, 22, 23, 24, 25]
        predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        wrong_count = metric.compute_single_event(actual, predicted)
        assert wrong_count == 5  # None of 5 positions found

    def test_compute_single_event_partial(self):
        """Test computing wrong count with partial matches."""
        metric = WrongPredictionsMetric(k=20)

        # 3 found (1, 5, 10), 2 missed (25, 30)
        actual = [1, 5, 10, 25, 30]
        predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        wrong_count = metric.compute_single_event(actual, predicted)
        assert wrong_count == 2

    def test_compute_single_event_respects_k(self):
        """Test that only top-k predictions are considered."""
        metric = WrongPredictionsMetric(k=5)

        actual = [1, 5, 10, 15, 20]
        # Position 10 is at index 9, beyond top-5
        predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

        wrong_count = metric.compute_single_event(actual, predicted)
        # Only 1 and 5 are in top-5, so 3 are wrong
        assert wrong_count == 3

    def test_compute_batch(self, sample_holdout_data):
        """Test batch computation of wrong counts."""
        metric = WrongPredictionsMetric(k=20)

        # Create predictions where first 20 positions are always predicted
        predictions = [[i for i in range(1, 21)] for _ in range(len(sample_holdout_data))]

        wrong_counts = metric.compute_batch(sample_holdout_data, predictions)

        assert len(wrong_counts) == len(sample_holdout_data)
        assert all(isinstance(w, (int, np.integer)) for w in wrong_counts)

        # For this data, positions cycle through patterns:
        # Pattern 0-3: All positions 1-19 (in top-20), wrong_count=0
        # Pattern 4: [20, 21, 22, 23, 24], 4 positions > 20, wrong_count=4
        # So 40 events with 0 wrong, 10 events with 4 wrong
        assert np.sum(wrong_counts == 0) == 40
        assert np.sum(wrong_counts == 4) == 10

    def test_compute_batch_mismatched_length(self, sample_holdout_data):
        """Test that batch computation fails with mismatched lengths."""
        metric = WrongPredictionsMetric(k=20)

        # Wrong number of predictions
        predictions = [[1, 2, 3] for _ in range(10)]

        with pytest.raises(ValueError, match="Number of predictions.*must match"):
            metric.compute_batch(sample_holdout_data, predictions)

    def test_compute_batch_missing_columns(self, sample_holdout_data):
        """Test that batch computation fails with missing columns."""
        metric = WrongPredictionsMetric(k=20)

        # Remove required column
        df = sample_holdout_data.drop(columns=['q_1'])
        predictions = [[1, 2, 3] for _ in range(len(df))]

        with pytest.raises(ValueError, match="Missing required columns"):
            metric.compute_batch(df, predictions)

    def test_compute_distribution(self):
        """Test computing distribution of wrong counts."""
        metric = WrongPredictionsMetric(k=20)

        # Sample wrong counts: 10 events with 0 wrong, 5 with 1 wrong, etc.
        wrong_counts = np.array([0]*10 + [1]*5 + [2]*3 + [3]*2 + [4]*1 + [5]*1)

        distribution = metric.compute_distribution(wrong_counts)

        assert len(distribution) == 6
        assert distribution[0]['count'] == 10
        assert distribution[0]['percentage'] == pytest.approx(45.45, rel=0.01)
        assert distribution[1]['count'] == 5
        assert distribution[1]['percentage'] == pytest.approx(22.73, rel=0.01)
        assert distribution[5]['count'] == 1
        assert distribution[5]['percentage'] == pytest.approx(4.55, rel=0.01)

    def test_compute_distribution_empty(self):
        """Test computing distribution with empty array."""
        metric = WrongPredictionsMetric(k=20)

        wrong_counts = np.array([])
        distribution = metric.compute_distribution(wrong_counts)

        assert distribution == {}

    def test_format_distribution_summary(self):
        """Test formatting distribution summary in user-specified format."""
        metric = WrongPredictionsMetric(k=20)

        distribution = {
            0: {'count': 36, 'percentage': 3.60},
            1: {'count': 166, 'percentage': 16.60},
            2: {'count': 322, 'percentage': 32.20},
            3: {'count': 313, 'percentage': 31.30},
            4: {'count': 146, 'percentage': 14.60},
            5: {'count': 17, 'percentage': 1.70},
        }

        summary = metric.format_distribution_summary(distribution, 1000, "Test Name")

        # Check that summary contains expected elements
        assert "HOLDOUT TEST SUMMARY - 1000 Events" in summary
        assert '"Test Name"' in summary
        assert "0 wrong: 36 events (3.60%)" in summary
        assert "All 5 actual values in top-20" in summary
        assert "5 wrong: 17 events (1.70%)" in summary
        assert "0 of 5 actual values in top-20" in summary


class TestMetricsHelpers:
    """Test helper metric functions."""

    def test_compute_top_k_accuracy(self):
        """Test top-k accuracy computation."""
        actual = [1, 5, 10, 15, 20]
        predicted = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        # 5 out of 5 found
        accuracy = compute_top_k_accuracy(actual, predicted, k=20)
        assert accuracy == 1.0

        # 3 out of 5 found (1, 5, 10)
        accuracy = compute_top_k_accuracy(actual, predicted, k=6)
        assert accuracy == 0.6

    def test_compute_recall_at_k(self):
        """Test recall@k computation."""
        actual = [1, 5, 10, 15, 20]
        predicted = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        # All 5 found in top-20
        recall = compute_recall_at_k(actual, predicted, k=20)
        assert recall == 1.0

        # 3 out of 5 found in top-6
        recall = compute_recall_at_k(actual, predicted, k=6)
        assert recall == 0.6

        # 0 out of 5 found in top-0
        recall = compute_recall_at_k(actual, predicted, k=0)
        assert recall == 0.0

    def test_compute_precision_at_k(self):
        """Test precision@k computation."""
        actual = [1, 5, 10, 15, 20]
        predicted = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        # 5 out of 20 predictions correct
        precision = compute_precision_at_k(actual, predicted, k=20)
        assert precision == 0.25

        # 3 out of 6 predictions correct
        precision = compute_precision_at_k(actual, predicted, k=6)
        assert precision == 0.5


class TestHoldoutTestRunner:
    """Test HoldoutTestRunner class."""

    def test_initialization(self, mock_ranker):
        """Test runner initialization."""
        runner = HoldoutTestRunner(mock_ranker, k=20)

        assert runner.ranker == mock_ranker
        assert runner.k == 20
        assert runner.collect_timing is True
        assert runner.collect_model_internals is True

    def test_initialization_custom_params(self, mock_ranker):
        """Test runner initialization with custom parameters."""
        runner = HoldoutTestRunner(
            mock_ranker,
            k=10,
            position_columns=['p1', 'p2', 'p3', 'p4', 'p5'],
            collect_timing=False,
            collect_model_internals=False
        )

        assert runner.k == 10
        assert runner.position_columns == ['p1', 'p2', 'p3', 'p4', 'p5']
        assert runner.collect_timing is False
        assert runner.collect_model_internals is False

    def test_run_holdout_test_basic(self, mock_ranker, sample_holdout_data):
        """Test basic holdout test execution."""
        runner = HoldoutTestRunner(mock_ranker, k=20, collect_timing=False)

        results = runner.run_holdout_test(sample_holdout_data, test_name="Test Run")

        # Check result structure
        assert 'test_name' in results
        assert 'test_metadata' in results
        assert 'per_event_metrics' in results
        assert 'timing_info' in results
        assert 'model_internals' in results
        assert 'summary' in results

        assert results['test_name'] == "Test Run"
        assert len(results['per_event_metrics']) == len(sample_holdout_data)

    def test_run_holdout_test_validates_columns(self, mock_ranker):
        """Test that runner validates required columns."""
        runner = HoldoutTestRunner(mock_ranker, k=20)

        # Data missing q_1 column
        bad_data = pd.DataFrame({
            'event_id': [1, 2, 3],
            'q_2': [2, 3, 4],
            'q_3': [3, 4, 5],
            'q_4': [4, 5, 6],
            'q_5': [5, 6, 7],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            runner.run_holdout_test(bad_data)

    def test_run_holdout_test_collects_metrics(self, mock_ranker, sample_holdout_data):
        """Test that runner collects per-event metrics."""
        runner = HoldoutTestRunner(mock_ranker, k=20, collect_timing=False)

        results = runner.run_holdout_test(sample_holdout_data)

        per_event_metrics = results['per_event_metrics']

        # Check that each event has expected metrics
        for event_metrics in per_event_metrics:
            assert 'event_index' in event_metrics
            assert 'actual_positions' in event_metrics
            assert 'predicted_top_k' in event_metrics
            assert 'wrong_count' in event_metrics
            assert 'recall_at_k' in event_metrics
            assert 'precision_at_k' in event_metrics
            assert 'accuracy' in event_metrics
            assert 'found_positions' in event_metrics
            assert 'missed_positions' in event_metrics

    def test_run_holdout_test_summary_statistics(self, mock_ranker, sample_holdout_data):
        """Test that runner computes summary statistics."""
        runner = HoldoutTestRunner(mock_ranker, k=20, collect_timing=False)

        results = runner.run_holdout_test(sample_holdout_data)

        summary = results['summary']

        assert 'total_events' in summary
        assert 'wrong_predictions_distribution' in summary
        assert 'average_recall_at_k' in summary
        assert 'average_precision_at_k' in summary
        assert 'average_accuracy' in summary

        assert summary['total_events'] == len(sample_holdout_data)

    def test_save_results(self, mock_ranker, sample_holdout_data):
        """Test saving holdout test results."""
        runner = HoldoutTestRunner(mock_ranker, k=20, collect_timing=False)

        results = runner.run_holdout_test(sample_holdout_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            paths = runner.save_results(results, output_dir, prefix='test')

            # Check that files were created
            assert 'summary' in paths
            assert 'per_event_metrics' in paths
            assert 'per_event_metrics_csv' in paths

            assert paths['summary'].exists()
            assert paths['per_event_metrics'].exists()
            assert paths['per_event_metrics_csv'].exists()

            # Check CSV can be loaded
            df = pd.read_csv(paths['per_event_metrics_csv'])
            assert len(df) == len(sample_holdout_data)


class TestReportGenerator:
    """Test ReportGenerator class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return {
            'test_name': 'Sample Test',
            'test_metadata': {
                'ranker_type': 'MockRanker',
                'num_events': 1000,
                'top_k': 20,
            },
            'summary': {
                'total_events': 1000,
                'wrong_predictions_distribution': {
                    0: {'count': 36, 'percentage': 3.60},
                    1: {'count': 166, 'percentage': 16.60},
                    2: {'count': 322, 'percentage': 32.20},
                    3: {'count': 313, 'percentage': 31.30},
                    4: {'count': 146, 'percentage': 14.60},
                    5: {'count': 17, 'percentage': 1.70},
                },
                'average_recall_at_k': 0.75,
                'average_precision_at_k': 0.1875,
                'average_accuracy': 0.75,
                'std_recall_at_k': 0.15,
                'std_precision_at_k': 0.0375,
                'std_accuracy': 0.15,
            },
            'timing_info': {
                'total_time_seconds': 10.5,
                'average_time_per_event_ms': 10.5,
            },
            'model_internals': {},
            'per_event_metrics': [],
        }

    def test_initialization(self, sample_results):
        """Test generator initialization."""
        generator = ReportGenerator(sample_results, k=20)

        assert generator.results == sample_results
        assert generator.k == 20

    def test_generate_summary_report(self, sample_results):
        """Test generating summary report."""
        generator = ReportGenerator(sample_results, k=20)

        summary = generator.generate_summary_report()

        # Check format
        assert "HOLDOUT TEST SUMMARY - 1000 Events" in summary
        assert '"Sample Test"' in summary
        assert "0 wrong: 36 events (3.60%)" in summary
        assert "5 wrong: 17 events (1.70%)" in summary
        assert "All 5 actual values in top-20" in summary

    def test_generate_detailed_report(self, sample_results):
        """Test generating detailed report."""
        generator = ReportGenerator(sample_results, k=20)

        report = generator.generate_detailed_report()

        # Check sections are present
        assert "DETAILED HOLDOUT TEST REPORT" in report
        assert "TEST METADATA" in report
        assert "AGGREGATE METRICS" in report
        assert "TIMING ANALYSIS" in report

    def test_save_reports(self, sample_results):
        """Test saving reports to files."""
        generator = ReportGenerator(sample_results, k=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            paths = generator.save_reports(output_dir, prefix='test', include_event_analysis=False)

            # Check that files were created
            assert 'summary' in paths
            assert 'detailed' in paths

            assert paths['summary'].exists()
            assert paths['detailed'].exists()

            # Check that files contain text
            summary_text = paths['summary'].read_text(encoding='utf-8')
            assert "HOLDOUT TEST SUMMARY" in summary_text

    def test_compare_rankers(self, sample_results):
        """Test comparing multiple rankers."""
        # Create two similar results with different distributions
        results1 = sample_results.copy()
        results2 = sample_results.copy()
        results2['summary'] = {
            **results2['summary'],
            'wrong_predictions_distribution': {
                0: {'count': 50, 'percentage': 5.00},
                1: {'count': 200, 'percentage': 20.00},
                2: {'count': 300, 'percentage': 30.00},
                3: {'count': 300, 'percentage': 30.00},
                4: {'count': 100, 'percentage': 10.00},
                5: {'count': 50, 'percentage': 5.00},
            },
        }

        comparison = compare_rankers(
            [results1, results2],
            ['Ranker A', 'Ranker B'],
            k=20
        )

        # Check format
        assert "RANKER COMPARISON REPORT" in comparison
        assert "WRONG PREDICTIONS DISTRIBUTION" in comparison
        assert "Ranker A" in comparison
        assert "Ranker B" in comparison

    def test_compare_rankers_mismatched_length(self, sample_results):
        """Test that comparing rankers fails with mismatched lengths."""
        with pytest.raises(ValueError, match="must have same length"):
            compare_rankers([sample_results], ['Ranker A', 'Ranker B'])


class TestIntegration:
    """Integration tests for evaluation module."""

    def test_full_workflow(self, mock_ranker, sample_holdout_data):
        """Test full workflow: run test, generate reports, save."""
        # Run holdout test
        runner = HoldoutTestRunner(mock_ranker, k=20, collect_timing=False)
        results = runner.run_holdout_test(sample_holdout_data, test_name="Integration Test")

        # Generate reports
        generator = ReportGenerator(results, k=20)
        summary = generator.generate_summary_report()
        detailed = generator.generate_detailed_report()

        # Check that reports were generated
        assert len(summary) > 0
        assert len(detailed) > 0

        # Save everything
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save results
            result_paths = runner.save_results(results, output_dir, prefix='integration')

            # Save reports
            report_paths = generator.save_reports(output_dir, prefix='integration', include_event_analysis=False)

            # Check that all files exist
            for path in result_paths.values():
                assert path.exists()

            for path in report_paths.values():
                assert path.exists()
