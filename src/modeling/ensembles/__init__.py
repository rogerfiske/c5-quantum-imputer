"""
Ensemble Methods for Quantum State Prediction

This module contains ensemble methods for combining multiple ranker predictions
to improve prediction robustness and accuracy.

Epic 9B implements simple ensemble approaches that aggregate predictions from
multiple trained models (LGBM, SetTransformer, GNN) to leverage their complementary
strengths.

Classes:
    SimpleEnsemble: Equal-weighted or confidence-weighted ensemble
    RangeAwareBiasCorrection: Range-aware bias correction layer

Author: BMad Dev Agent (James)
Date: 2025-10-17
Epic: Epic 9B - Ensemble & Bias Correction
"""

from .simple_ensemble import SimpleEnsemble
from .bias_correction import RangeAwareBiasCorrection

__all__ = ['SimpleEnsemble', 'RangeAwareBiasCorrection']
