"""
Ranking Models Sub-Module

This sub-module contains implementations of ranking model families
for predicting the next quantum state.

Models:
1. BaseRanker (base_ranker.py)
   - Abstract base class for all ranking models

2. Frequency Baselines (frequency_ranker.py)
   - Simple statistical baselines using historical frequencies
   - Methods: cumulative, EMA, bigram co-occurrence

3. LightGBM/XGBoost Ranker (lgbm_ranker.py) [Story 3.3 - Implemented]
   - Gradient-boosted decision tree ranking models using LightGBM

4. Set Transformer Ranker (deepsets_ranker.py) [Story 3.4 - Implemented]
   - Attention-based set processing with ISAB and PMA

5. Graph Neural Network (gnn_ranker.py) [Story 3.5 - Implemented]
   - GNN operating on the C₃₉ cyclic graph structure with GAT layers

Author: BMad Dev Agent (James)
Date: 2025-10-14
Epic: Epic 3 - Individual Ranker Implementation
"""

from .base_ranker import BaseRanker
from .frequency_ranker import FrequencyRanker
from .lgbm_ranker import LGBMRanker
from .deepsets_ranker import SetTransformerRanker
from .gnn_ranker import GNNRanker

__all__ = [
    "BaseRanker",
    "FrequencyRanker",
    "LGBMRanker",
    "SetTransformerRanker",
    "GNNRanker",
]
