"""
Position-wise Isotonic Calibration (Repo-Style Pack)
====================================================
Path: src/modeling/calibration/position_isotonic.py

Provides:
    - `PositionIsotonicCalibrator`: per-position isotonic calibration
    - CLI to fit/save calibrators and to apply calibration to score matrices
    - Utilities to compute Recall@20 and middle-zone breakdown

Expected data:
    - Scores frame with columns: ["event_id"] + ["p1", ..., "p39"]
    - Labels frame with columns: ["event_id"] + ["y1", ..., "y39"]

Outputs:
    - Calibrated probabilities parquet with same shape as scores (p1..p39)
    - Optional Top-20 indices per event (top20_1..top20_20)
    - Saved calibrator object (.joblib)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationPackConfig:
    """
    Configuration for position-wise isotonic calibration operations.

    Args:
        normalize: Whether to renormalize calibrated outputs to sum to 1 per event.
    """

    normalize: bool = True


class PositionIsotonicCalibrator:
    """
    Per-position isotonic calibration for 39 positions.

    Methods:
        fit(scores, labels): Fit isotonic models per position.
        transform(scores): Apply calibration; optionally renormalize rows.
        save(path): Persist calibrator via joblib.
        load(path): Load a persisted calibrator.
    """

    def __init__(self, config: Optional[CalibrationPackConfig] = None) -> None:
        self.config = config or CalibrationPackConfig()
        self.models: Dict[int, IsotonicRegression] = {}
        self._fitted: bool = False

    def fit(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> "PositionIsotonicCalibrator":
        """
        Fit per-position isotonic models.

        Args:
            scores: Array of shape (n_events, 39), raw scores.
            labels: Array of shape (n_events, 39), binary 0/1.

        Returns:
            Self.
        """
        n_pos = scores.shape[1]
        for j in range(n_pos):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(scores[:, j], labels[:, j])
            self.models[j] = iso
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw scores.

        Args:
            scores: Array of shape (n_events, 39).

        Returns:
            Calibrated probabilities of shape (n_events, 39).
        """
        assert self._fitted, "Calibrator not fitted. Call fit() first."
        out = np.zeros_like(scores, dtype=float)
        for j, iso in self.models.items():
            out[:, j] = iso.transform(scores[:, j])

        if self.config.normalize:
            row_sum = np.sum(out, axis=1, keepdims=True)
            row_sum = np.where(row_sum <= 0.0, 1.0, row_sum)
            out = out / row_sum
        return out

    def save(self, path: str) -> None:
        """
        Save the calibrator to disk using joblib.

        Args:
            path: Destination file path (.joblib).
        """
        joblib.dump({"config": self.config, "models": self.models}, path)

    @staticmethod
    def load(path: str) -> "PositionIsotonicCalibrator":
        """
        Load a saved calibrator.

        Args:
            path: Source file path (.joblib).

        Returns:
            Loaded calibrator instance.
        """
        payload = joblib.load(path)
        cal = PositionIsotonicCalibrator(payload["config"])
        cal.models = payload["models"]
        cal._fitted = True
        return cal


# ----------------
# Helper functions
# ----------------


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


def recall_at_20(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Recall@20 = average hits / 5, expressed as percentage.

    Args:
        y_true: Binary labels (n_events, 39).
        p: Probabilities (n_events, 39).

    Returns:
        Recall@20 as percentage.
    """
    hits = _hits_at_k(y_true, p, k=20)
    return float(np.mean(hits) / 5.0 * 100.0)


def distribution_breakdown(
    y_true: np.ndarray, p: np.ndarray, k: int = 20
) -> Dict[str, float]:
    """
    Compute distribution over {0–1, 2–3, 4–5 wrong}.

    Args:
        y_true: Binary labels (n_events, 39).
        p: Probabilities (n_events, 39).
        k: Top-k (default 20).

    Returns:
        Dictionary with percentage breakdown.
    """
    hits = _hits_at_k(y_true, p, k=k)
    b_01 = float(np.mean(np.isin(hits, [4, 5])) * 100.0)  # 0–1 wrong == 4–5 hits
    b_23 = float(np.mean(np.isin(hits, [2, 3])) * 100.0)
    b_45 = float(np.mean(np.isin(hits, [0, 1])) * 100.0)  # 4–5 wrong == 0–1 hits
    return {"0-1_wrong": b_01, "2-3_wrong": b_23, "4-5_wrong": b_45}


# ---
# CLI
# ---


def _cli() -> None:
    """
    Command-line interface.

    Actions:
        - fit: Fit and save calibrator, write calibrated parquet and optional top-20.
        - apply: Load calibrator and apply to scores.
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="Fit per-position calibrators")
    p_fit.add_argument(
        "--scores", required=True, type=str, help="Path to scores.parquet"
    )
    p_fit.add_argument(
        "--labels", required=True, type=str, help="Path to labels.parquet"
    )
    p_fit.add_argument(
        "--save-model", required=True, type=str, help="Output .joblib path"
    )
    p_fit.add_argument(
        "--out-calibrated", required=True, type=str, help="Output calibrated parquet"
    )
    p_fit.add_argument(
        "--out-topk", required=False, type=str, help="Optional Top-20 indices parquet"
    )
    p_fit.add_argument(
        "--normalize", action="store_true", help="Renormalize rows to sum to 1"
    )

    p_apply = sub.add_parser("apply", help="Apply saved calibrator")
    p_apply.add_argument(
        "--scores", required=True, type=str, help="Path to scores.parquet"
    )
    p_apply.add_argument(
        "--load-model", required=True, type=str, help="Path to saved .joblib"
    )
    p_apply.add_argument(
        "--out-calibrated", required=True, type=str, help="Output calibrated parquet"
    )

    args = parser.parse_args()

    if args.cmd == "fit":
        scores_df = pd.read_parquet(args.scores)
        labels_df = pd.read_parquet(args.labels)
        score_cols = [f"p{i}" for i in range(1, 40)]
        label_cols = [f"y{i}" for i in range(1, 40)]
        scores = scores_df[score_cols].to_numpy(dtype=float)
        labels = labels_df[label_cols].to_numpy(dtype=int)

        cal = PositionIsotonicCalibrator(
            CalibrationPackConfig(normalize=bool(args.normalize))
        )
        cal.fit(scores, labels)
        p_cal = cal.transform(scores)

        out_df = pd.DataFrame({"event_id": scores_df["event_id"]})
        for i in range(39):
            out_df[f"p{i+1}"] = p_cal[:, i]
        out_df.to_parquet(args.out_calibrated, index=False)

        if args.out_topk:
            idx = np.argsort(-p_cal, axis=1)[:, :20]
            topk_df = pd.DataFrame({"event_id": scores_df["event_id"]})
            for k in range(20):
                topk_df[f"top20_{k+1}"] = idx[:, k] + 1  # 1..39
            topk_df.to_parquet(args.out_topk, index=False)

        cal.save(args.save_model)

        # Print quick metrics if labels present
        r20 = recall_at_20(labels, p_cal)
        dist = distribution_breakdown(labels, p_cal, k=20)
        print(f"Recall@20: {r20:.2f}%")
        print("Distribution (%):", dist)

    elif args.cmd == "apply":
        scores_df = pd.read_parquet(args.scores)
        score_cols = [f"p{i}" for i in range(1, 40)]
        scores = scores_df[score_cols].to_numpy(dtype=float)

        cal = PositionIsotonicCalibrator.load(args.load_model)
        p_cal = cal.transform(scores)

        out_df = pd.DataFrame({"event_id": scores_df["event_id"]})
        for i in range(39):
            out_df[f"p{i+1}"] = p_cal[:, i]
        out_df.to_parquet(args.out_calibrated, index=False)


if __name__ == "__main__":
    _cli()

# -------------------------------
# requirements.txt (used versions)
# -------------------------------
# numpy==2.1.3
# pandas==2.2.3
# scikit-learn==1.5.2
# pyarrow==17.0.0
# joblib==1.4.2
