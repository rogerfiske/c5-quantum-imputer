"""
Extremizer Meta-Layer for C5 Quantum Lottery Imputer
----------------------------------------------------
Event-level sharpening/flattening of score vectors to reduce the proportion
of "2–3 wrong" outcomes while preserving or modestly improving Recall@20.

Features:
    - Per-position isotonic calibration to reduce chronic false positives
    - Event-level diagnostics (entropy, Gini, top-K mass, score gaps, HHI)
    - Middle-zone risk classifier (predicts if an event is likely to end with 2–3 hits)
    - Temperature-based reshaping (gamma) of probabilities: sharpen vs. flatten
    - Top-20 selection and evaluation utilities

Usage (CLI):
    python extremizer.py \
        --scores scores.parquet \
        --labels labels.parquet \
        --out predictions.parquet

Expected input formats:
    - scores.parquet: columns ["event_id"] + 39 columns "p1"..."p39" with raw scores
    - labels.parquet: columns ["event_id"] + 39 binary columns "y1"..."y39" (1 if position is a winner)

The tool will fit calibrators + classifier on the training fold inferred from event_id,
apply the extremizer, and write Top-20 selections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import argparse
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """
    Compute softmax with temperature.

    Args:
        x: Array of scores.
        temperature: Temperature parameter (lower -> sharper).
        axis: Axis for softmax operation.

    Returns:
        Softmax probabilities with the same shape as x.
    """
    z = x / max(temperature, 1e-8)
    z -= np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Shannon entropy for probability vectors.

    Args:
        p: Probability array that sums to 1 along `axis`.
        axis: Axis over which to compute entropy.
        eps: Numerical stability constant.

    Returns:
        Entropy values.
    """
    p_clip = np.clip(p, eps, 1.0)
    return -np.sum(p_clip * np.log(p_clip), axis=axis)


def gini_coefficient(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the Gini coefficient for probability vectors.

    Args:
        p: Probability array that sums to 1 along `axis`.
        axis: Axis over which to compute Gini.

    Returns:
        Gini values.
    """
    p_roll = np.moveaxis(p, axis, -1)
    sort_p = np.sort(p_roll, axis=-1)
    n = sort_p.shape[-1]
    cum = np.cumsum(sort_p, axis=-1)
    g = (n + 1 - 2 * np.sum(cum / cum[..., -1:], axis=-1)) / n
    return g


def topk_mass(p: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """
    Compute the total probability mass of the Top-k entries.

    Args:
        p: Probability array.
        k: Top-k.
        axis: Axis to consider.

    Returns:
        Sum of Top-k masses.
    """
    p_roll = np.moveaxis(p, axis, -1)
    sort_p = np.sort(p_roll, axis=-1)[:, ::-1]
    k = min(k, sort_p.shape[-1])
    return np.sum(sort_p[..., :k], axis=-1)


def hhi_index(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Herfindahl–Hirschman Index: sum of squared probabilities.

    Args:
        p: Probability array.
        axis: Axis to consider.

    Returns:
        HHI values.
    """
    return np.sum(p * p, axis=axis)


class PositionIsotonicCalibrator:
    """
    Per-position isotonic regressors to calibrate raw scores -> probabilities.
    """
    def __init__(self) -> None:
        self._models: Dict[int, IsotonicRegression] = {}

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PositionIsotonicCalibrator":
        """
        Fit per-position isotonic calibrators.

        Args:
            scores: Array shape (n_events, 39) of raw scores.
            labels: Array shape (n_events, 39) of binary labels.

        Returns:
            Self.
        """
        n_pos = scores.shape[1]
        for j in range(n_pos):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(scores[:, j], labels[:, j])
            self._models[j] = iso
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration.

        Args:
            scores: Array shape (n_events, 39).

        Returns:
            Calibrated probabilities in [0, 1].
        """
        out = np.zeros_like(scores, dtype=float)
        for j, iso in self._models.items():
            out[:, j] = iso.transform(scores[:, j])
        # Normalize to probability simplex to produce a distribution per event.
        out_sum = np.sum(out, axis=1, keepdims=True)
        out_sum = np.where(out_sum <= 0, 1.0, out_sum)
        return out / out_sum


@dataclass
class ExtremizerConfig:
    """
    Configuration for event-level temperature shaping.
    """
    gamma_sharp: float = 1.6
    gamma_flat: float = 0.8
    risk_threshold: float = 0.5  # if risk_{2-3} >= threshold -> flatten


class Extremizer(BaseEstimator, TransformerMixin):
    """
    Extremizer meta-layer:
      1) Calibrate per-position scores -> probabilities
      2) Compute event-level diagnostics
      3) Predict middle-zone risk
      4) Sharpen or flatten per-event probabilities with gamma
      5) Select Top-20
    """
    def __init__(self, config: Optional[ExtremizerConfig] = None) -> None:
        self.config = config or ExtremizerConfig()
        self.calibrator = PositionIsotonicCalibrator()
        self.clf = LogisticRegression(max_iter=200, n_jobs=None)
        self._fitted = False

    @staticmethod
    def _diagnostics(p: np.ndarray) -> np.ndarray:
        """
        Build diagnostic features from probability vectors.

        Args:
            p: Probabilities shape (n_events, 39).

        Returns:
            Diagnostics array shape (n_events, d_diag).
        """
        ent = entropy(p)
        gini = gini_coefficient(p)
        hhi = hhi_index(p)
        top1 = topk_mass(p, 1)
        top3 = topk_mass(p, 3)
        top5 = topk_mass(p, 5)

        # gaps between sorted top probabilities
        p_sorted = np.sort(p, axis=1)[:, ::-1]
        gaps = np.stack([p_sorted[:, i] - p_sorted[:, i + 1] for i in range(5)], axis=1)  # 5 gaps

        feats = np.column_stack([ent, gini, hhi, top1, top3, top5, gaps])
        return feats

    @staticmethod
    def _apply_gamma(p: np.ndarray, gamma: float) -> np.ndarray:
        """
        Raise probabilities to a power gamma and renormalize.

        Args:
            p: Probabilities (n_events, 39).
            gamma: Power factor (>1 sharpens, <1 flattens).

        Returns:
            Adjusted probabilities with same shape.
        """
        p_adj = np.power(np.clip(p, 1e-12, 1.0), gamma)
        p_adj /= np.sum(p_adj, axis=1, keepdims=True)
        return p_adj

    @staticmethod
    def _hits_at_k(y_true: np.ndarray, p: np.ndarray, k: int = 20) -> np.ndarray:
        """
        Compute number of true hits found within Top-k for each event.

        Args:
            y_true: Binary labels (n_events, 39) with exactly 5 ones per row.
            p: Probabilities (n_events, 39).
            k: Top-k.

        Returns:
            Integer array (n_events,) of hits within Top-k.
        """
        idx = np.argsort(-p, axis=1)[:, :k]
        rows = np.arange(p.shape[0])[:, None]
        return np.sum(y_true[rows, idx], axis=1)

    def fit(self, scores: np.ndarray, labels: np.ndarray, groups: Optional[np.ndarray] = None) -> "Extremizer":
        """
        Fit calibrator and middle-risk classifier via rolling groups (if provided).

        Args:
            scores: Raw scores (n_events, 39).
            labels: Binary labels (n_events, 39).
            groups: Optional grouping for time-based CV (e.g., event_id blocks).

        Returns:
            Self.
        """
        # Step 1: Fit calibrator on all training data (or could use CV averaging).
        self.calibrator.fit(scores, labels)
        p_cal = self.calibrator.transform(scores)

        # Step 2: Build risk labels: 1 if event hits in {2,3} within Top-20 using calibrated p.
        hits = self._hits_at_k(labels, p_cal, k=20)
        y_risk = np.isin(hits, [2, 3]).astype(int)

        X_diag = self._diagnostics(p_cal)

        # Step 3: Train classifier with grouped CV AUC report (optional).
        if groups is None:
            self.clf.fit(X_diag, y_risk)
        else:
            gkf = GroupKFold(n_splits=5)
            aucs: List[float] = []
            for tr, va in gkf.split(X_diag, y_risk, groups=groups):
                self.clf.fit(X_diag[tr], y_risk[tr])
                pr = self.clf.predict_proba(X_diag[va])[:, 1]
                try:
                    aucs.append(roc_auc_score(y_risk[va], pr))
                except ValueError:
                    pass
            # Refit on all
            self.clf.fit(X_diag, y_risk)

        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration + event-level reshaping.

        Args:
            scores: Raw scores (n_events, 39).

        Returns:
            Adjusted probabilities (n_events, 39).
        """
        assert self._fitted, "Call fit() before transform()."
        p = self.calibrator.transform(scores)
        X_diag = self._diagnostics(p)
        risk = self.clf.predict_proba(X_diag)[:, 1]

        # Map risk to gamma: low risk -> sharpen, high risk -> flatten.
        gamma = np.where(
            risk >= self.config.risk_threshold,
            self.config.gamma_flat,
            self.config.gamma_sharp,
        )
        # Apply per-event gamma.
        p_out = np.vstack([self._apply_gamma(p[i:i+1], gamma[i]) for i in range(p.shape[0])])
        return p_out

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convenience: fit then transform.

        Args:
            scores: Raw scores (n_events, 39).
            labels: Binary labels (n_events, 39).
            groups: Optional grouping for CV.

        Returns:
            Adjusted probabilities.
        """
        self.fit(scores, labels, groups)
        return self.transform(scores)

    @staticmethod
    def topk_indices(p: np.ndarray, k: int = 20) -> np.ndarray:
        """
        Indices of Top-k per event.

        Args:
            p: Probabilities (n_events, 39).
            k: Top-k.

        Returns:
            Integer indices (n_events, k).
        """
        return np.argsort(-p, axis=1)[:, :k]

    @staticmethod
    def distribution_breakdown(y_true: np.ndarray, p: np.ndarray, k: int = 20) -> dict:
        """
        Compute distribution over {0–1, 2–3, 4–5 wrong}.

        Args:
            y_true: Binary labels (n_events, 39).
            p: Probabilities (n_events, 39).
            k: Top-k.

        Returns:
            Dict with percentage breakdown.
        """
        hits = Extremizer._hits_at_k(y_true, p, k)
        n = len(hits)
        b_01 = float(np.mean(np.isin(hits, [4, 5])) * 100.0)  # 0–1 wrong == 4–5 hits
        b_23 = float(np.mean(np.isin(hits, [2, 3])) * 100.0)
        b_45 = float(np.mean(np.isin(hits, [0, 1])) * 100.0)  # 4–5 wrong == 0–1 hits
        return {"0-1_wrong": b_01, "2-3_wrong": b_23, "4-5_wrong": b_45}

    @staticmethod
    def recall_at_20(y_true: np.ndarray, p: np.ndarray) -> float:
        """
        Recall@20 = average hits / 5.

        Args:
            y_true: Binary labels (n_events, 39).
            p: Probabilities (n_events, 39).

        Returns:
            Recall@20 as a percentage.
        """
        hits = Extremizer._hits_at_k(y_true, p, k=20)
        return float(np.mean(hits) / 5.0 * 100.0)


def _cli() -> None:
    """
    Command-line interface for the extremizer module.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, required=True, help="Path to scores.parquet")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels.parquet")
    parser.add_argument("--out", type=str, required=True, help="Output parquet for Top-20 indices")
    parser.add_argument("--risk-threshold", type=float, default=0.5, help="Threshold for flattening vs sharpening")
    parser.add_argument("--gamma-sharp", type=float, default=1.6, help="Gamma for sharpening (>1)")
    parser.add_argument("--gamma-flat", type=float, default=0.8, help="Gamma for flattening (<1)")
    args = parser.parse_args()

    scores_df = pd.read_parquet(args.scores)
    labels_df = pd.read_parquet(args.labels)

    # Expect columns p1..p39 and y1..y39
    score_cols = [f"p{i}" for i in range(1, 40)]
    label_cols = [f"y{i}" for i in range(1, 40)]
    scores = scores_df[score_cols].to_numpy(dtype=float)
    labels = labels_df[label_cols].to_numpy(dtype=int)

    cfg = ExtremizerConfig(
        gamma_sharp=args.gamma_sharp,
        gamma_flat=args.gamma_flat,
        risk_threshold=args.risk_threshold,
    )
    ex = Extremizer(cfg)
    p_adj = ex.fit_transform(scores, labels, groups=scores_df.get("event_id", None))

    # Write Top-20 predictions
    idx_top20 = Extremizer.topk_indices(p_adj, k=20)
    out = pd.DataFrame({
        "event_id": scores_df["event_id"],
        **{f"top20_{i+1}": idx_top20[:, i] + 1 for i in range(20)}  # +1 -> positions 1..39
    })
    out.to_parquet(args.out, index=False)

    # Optional: print quick metrics if labels provided.
    r20 = Extremizer.recall_at_20(labels, p_adj)
    dist = Extremizer.distribution_breakdown(labels, p_adj, k=20)
    print(f"Recall@20: {r20:.2f}%")
    print("Distribution (%):", dist)


if __name__ == "__main__":
    _cli()

# -------------------------------
# requirements.txt (used versions)
# -------------------------------
# numpy==2.1.3
# pandas==2.2.3
# scikit-learn==1.5.2
# pyarrow==17.0.0
