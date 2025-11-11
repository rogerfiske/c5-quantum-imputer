"""
Evaluation Module

This module contains tools for evaluating quantum state prediction models
on holdout test data. It includes:

1. Metrics (metrics.py):
   - Wrong predictions metric (0-5 wrong distribution)
   - Top-k accuracy metrics
   - Prediction quality analysis

2. Holdout Test Runner (holdout_test.py):
   - Run holdout tests on trained ranker models
   - Collect detailed metrics at each prediction step
   - Track computational performance
   - Save metrics for post-processing

3. Report Generator (report_generator.py):
   - Generate summary reports in specified format
   - Visualize wrong predictions distribution
   - Export detailed metrics for analysis

Author: BMad Dev Agent (James)
Date: 2025-10-14
Epic: Epic 4 - Evaluation and Reporting
"""

from .metrics import WrongPredictionsMetric, compute_top_k_accuracy
from .holdout_test import HoldoutTestRunner
from .report_generator import ReportGenerator
from .shap_attribution import LGBMShapExplainer, compute_position_specific_importance

__all__ = [
    'WrongPredictionsMetric',
    'compute_top_k_accuracy',
    'HoldoutTestRunner',
    'ReportGenerator',
    'LGBMShapExplainer',
    'compute_position_specific_importance',
]
