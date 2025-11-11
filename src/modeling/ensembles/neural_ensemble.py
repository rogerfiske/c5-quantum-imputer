"""
Neural Pattern Ensemble - Combines Transformer and SetTransformer

Ensemble strategy: Simple averaging of predictions from both models

Author: BMad Dev Agent (James)
Date: 2025-11-04
"""

import numpy as np
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.modeling.rankers.neural_pattern_ranker_simple import SimpleNeuralPatternRanker
from src.modeling.rankers.neural_pattern_settransformer import NeuralPatternSetTransformerRanker

logger = logging.getLogger(__name__)


class NeuralEnsembleRanker:
    """
    Ensemble combining Transformer and SetTransformer for pattern-based ranking.

    Strategy:
    - Load both trained models
    - Average predictions (simple mean)
    - Return ensemble scores

    Expected performance: 52.7-53.0% (both individual models at ~52.6%)
    """

    def __init__(
        self,
        transformer_path: str = "production/models/neural_pattern/neural_pattern_best.pth",
        settransformer_path: str = "production/models/neural_pattern/settransformer_best.pth",
        vocab_size: int = 158,
        device: Optional[str] = None
    ):
        """
        Initialize ensemble with both models.

        Args:
            transformer_path: Path to trained Transformer model
            settransformer_path: Path to trained SetTransformer model
            vocab_size: Vocabulary size (default 158 = 93 patterns + padding + special tokens)
            device: Device to use (default: auto-detect)
        """
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing Neural Ensemble on: {self.device}")

        # Initialize Transformer
        logger.info("Loading Transformer model...")
        self.transformer = SimpleNeuralPatternRanker(
            vocab_size=vocab_size,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=256,
            dropout=0.1,
            max_seq_len=50,
            device=self.device
        )

        # Load Transformer weights
        transformer_path = Path(transformer_path)
        if transformer_path.exists():
            checkpoint = torch.load(transformer_path, map_location=self.device, weights_only=False)
            self.transformer.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded Transformer from {transformer_path}")
        else:
            raise FileNotFoundError(f"Transformer model not found: {transformer_path}")

        # Initialize SetTransformer
        logger.info("Loading SetTransformer model...")
        self.settransformer = NeuralPatternSetTransformerRanker(
            vocab_size=vocab_size,
            d_model=64,
            num_heads=4,
            num_inds=16,
            num_layers=2,
            num_seeds=1,
            ff_dim=256,
            dropout=0.1,
            device=self.device
        )

        # Load SetTransformer weights
        settransformer_path = Path(settransformer_path)
        if settransformer_path.exists():
            checkpoint = torch.load(settransformer_path, map_location=self.device, weights_only=False)
            self.settransformer.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded SetTransformer from {settransformer_path}")
        else:
            raise FileNotFoundError(f"SetTransformer model not found: {settransformer_path}")

        # Set both models to eval mode
        self.transformer.model.eval()
        self.settransformer.model.eval()

        logger.info("Neural Ensemble ready!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict scores for pattern sequences (batch of positions).

        Args:
            X: Pattern sequences (n_samples, seq_len)

        Returns:
            Ensemble scores (n_samples,)
        """
        with torch.no_grad():
            # Transformer predictions
            X_tensor = torch.from_numpy(X).long().to(self.device)
            transformer_logits = self.transformer.model(X_tensor)
            transformer_probs = torch.sigmoid(transformer_logits).cpu().numpy()

            # SetTransformer predictions
            settransformer_logits = self.settransformer.model(X_tensor)
            settransformer_probs = torch.sigmoid(settransformer_logits.squeeze(-1)).cpu().numpy()

            # Simple average
            ensemble_scores = (transformer_probs + settransformer_probs) / 2.0

        return ensemble_scores

    def predict_event(self, X_event: np.ndarray) -> np.ndarray:
        """
        Predict scores for a single event (39 positions).

        Args:
            X_event: Pattern sequences for one event (39, seq_len)

        Returns:
            Scores for 39 positions (39,)
        """
        return self.predict(X_event)

    def evaluate_recall_at_k(
        self,
        X_val_events: List[np.ndarray],
        y_val_events: List[np.ndarray],
        k: int = 20
    ) -> Dict[str, Any]:
        """
        Evaluate recall@k on validation events.

        Args:
            X_val_events: List of event pattern sequences (each: 39, seq_len)
            y_val_events: List of event binary labels (each: 39,)
            k: Number of top predictions to consider

        Returns:
            Dictionary with evaluation metrics
        """
        recalls = []
        wrong_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        detailed_results = []

        for event_idx, (X_event, y_event) in enumerate(zip(X_val_events, y_val_events)):
            # Get predictions
            scores = self.predict_event(X_event)

            # Top-k predictions
            top_k_indices = np.argsort(scores)[::-1][:k]

            # Count hits
            hits = y_event[top_k_indices].sum()
            total_winners = y_event.sum()
            recall = hits / total_winners if total_winners > 0 else 0.0
            recalls.append(recall)

            # Count wrong predictions (5 - hits)
            num_wrong = int(total_winners - hits)
            wrong_counts[num_wrong] += 1

            # Store detailed result
            detailed_results.append({
                "event_index": int(event_idx),
                "recall": float(recall),
                "num_wrong": int(num_wrong),
                "top_k_predictions": top_k_indices.tolist()
            })

        # Calculate metrics
        mean_recall = np.mean(recalls)

        # Calculate percentages for wrong distribution
        total_events = len(X_val_events)
        wrong_distribution = {}
        for wrong, count in wrong_counts.items():
            wrong_distribution[str(wrong)] = {
                "count": int(count),
                "percentage": float((count / total_events) * 100)
            }

        return {
            "mean_recall_at_20": float(mean_recall),
            "num_events": int(total_events),
            "wrong_distribution": wrong_distribution,
            "detailed_results": detailed_results
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about ensemble components."""
        return {
            "ensemble_type": "simple_average",
            "models": [
                {
                    "type": "Transformer",
                    "architecture": "sequence-based with positional encoding",
                    "parameters": sum(p.numel() for p in self.transformer.model.parameters())
                },
                {
                    "type": "SetTransformer",
                    "architecture": "permutation-invariant with ISAB/PMA",
                    "parameters": sum(p.numel() for p in self.settransformer.model.parameters())
                }
            ],
            "device": str(self.device)
        }
