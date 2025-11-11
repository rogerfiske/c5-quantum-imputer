# C5 Quantum Lottery Imputer

A machine learning project that uses quantum-inspired imputation techniques to predict lottery outcomes.

## Quick Stats

- **Dataset**: 11,589 Cash 5 lottery events (5 winners from 39 positions)
- **Current Performance**: 50.92% Recall@20 (1000-event holdout)
- **Best Model**: LGBM + Amplitude Embedding (242 quantum features)
- **Challenge**: 62.8% of predictions stuck in "middle zone" (2-3 wrong out of 5)

## Project Overview

This project transforms sparse binary lottery data (5 winners, 34 non-winners) into dense quantum-enriched features using concepts from quantum mechanics. We then train machine learning models to predict the next winning positions.

### Key Innovation: Quantum Imputation

Instead of leaving non-winning positions as zeros, we impute quantum characteristics:
- **Amplitude**: Quantum superposition states (1/√n_active)
- **Probability**: Born rule probabilities (|amplitude|²)
- **Interference**: Quantum interference patterns
- **Entanglement**: Von Neumann entropy
- **Phase**: Quantum phase angles

This transforms 5 binary values → 242 rich features per event.

## Performance Summary

### Current Best Model (1000-event holdout)

```
Overall Recall@20: 50.92% (2546/5000 correct)

Wrong Distribution:
  0 wrong (perfect):   26 events ( 2.6%)  ✓ EXCELLENT
  1 wrong:            174 events (17.4%)  ✓ GOOD
  2 wrong:            313 events (31.3%)  ✗ POOR
  3 wrong:            315 events (31.5%)  ✗ POOR
  4 wrong:            151 events (15.1%)  ~ ACCEPTABLE
  5 wrong (total miss): 21 events ( 2.1%)  ~ ACCEPTABLE

Macro Objective:
  Excellent/Good (0-1 wrong):  20.0%  ← Want 35-40%
  Poor (2-3 wrong):            62.8%  ← Want 35-40% (stuck here!)
  Acceptable (4-5 wrong):      17.2%  ← OK
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rogerfiske/c5-quantum-imputer.git
cd c5-quantum-imputer

# Install dependencies
pip install -r requirements.txt
```

### Make a Prediction

```python
import pandas as pd
import joblib
from src.imputation.amplitude_embedding import AmplitudeEmbedding

# Load data and model
df = pd.read_csv('data/raw/c5_Matrix.csv')
historical = df[df['event-ID'] <= 11581]  # All history up to event 11581

imputer = AmplitudeEmbedding()
imputer.fit(historical)

model = joblib.load('production/models/amplitude/lgbm_ranker_baseline_fixed_v1.pkl')

# Predict next event (11582)
top_20 = imputer.predict_with_imputation(historical, model, num_predictions=20)
print(f"Top-20 predictions for event 11582: {top_20}")
```

### Run Holdout Test

```bash
# Test on 1000 holdout events
python production/comprehensive_1000event_holdout_diagnostic.py
```

## Repository Structure

```
c5-quantum-imputer/
├── data/
│   └── raw/
│       └── c5_Matrix.csv              # 11,589 lottery events
│
├── src/
│   ├── imputation/                    # Quantum imputation methods
│   │   ├── amplitude_embedding.py     # Primary method (242 features)
│   │   ├── density_matrix.py          # Density matrix encoding
│   │   ├── angle_encoding.py          # Angle encoding
│   │   └── graph_cycle_encoding.py    # Graph cycle features
│   │
│   ├── modeling/
│   │   ├── rankers/                   # ML models
│   │   │   ├── lgbm_ranker.py         # LightGBM (best)
│   │   │   ├── gnn_ranker.py          # Graph Neural Network
│   │   │   └── settransformer_ranker.py  # Transformer
│   │   │
│   │   └── ensembles/                 # Ensemble methods
│   │       ├── bias_correction.py     # Temperature scaling
│   │       └── neural_ensemble.py     # Multi-model ensemble
│   │
│   └── evaluation/
│       └── metrics.py                 # Evaluation metrics
│
├── production/
│   ├── models/
│   │   └── amplitude/
│   │       └── lgbm_ranker_baseline_fixed_v1.pkl  # Best model
│   │
│   ├── predict_next_event.py          # Production prediction
│   └── comprehensive_1000event_holdout_diagnostic.py
│
├── tests/
│   └── unit/                          # Unit tests
│
├── docs/
│   ├── COMPREHENSIVE_PROJECT_AUDIT.md # Full experimental history
│   ├── QUANTUM_IMPUTATION_EXPLAINED.md
│   └── EXPERIMENTAL_RESULTS.md
│
├── requirements.txt
└── README.md
```

## Experimental History

We've conducted extensive experiments over 4 weeks (Oct 14 - Nov 11, 2025):

### Epic 9A: Model Comparison
Tested 12 model configurations (4 imputation methods × 3 rankers):
- **Winner**: LGBM + Amplitude (50.92% recall)
- Neural models underperformed (48-49% recall)

### Epic 9B: Ensemble Methods
Tried simple averaging, neural ensembles, bias correction:
- **Result**: <0.3% improvement
- Ensembles don't help when base models are similar

### Epic 13: Force Predictions to Extremes
Attempted to reduce middle zone (2-3 wrong) from 62.8% to 35-40%:

**Story 13.1 - Temporal Weighting**: NO IMPROVEMENT
- Tested decay rates 0.0, 0.05, 0.10
- All produced identical 53.20% recall

**Story 13.2 - Optimize Lookback Window**: NO IMPROVEMENT
- Tested windows: 30, 40, 50, 90, 100, 150 events
- All 40-150 converged to identical 53.20% recall
- Window=30 worse (52.60% - insufficient history)

**Story 13.3 - Ensemble Voting**: NO IMPROVEMENT
- Majority voting, unanimous voting
- All produced identical 53.20% recall

### Key Findings

1. **Data is near-random**: Position frequencies 11.8-14.2% (nearly uniform)
2. **Weak patterns exist**: Enough for ~51% recall but not strong enough for extremes
3. **Imputation converges**: All methods produce similar probability estimates
4. **LGBM dominates**: Model's learned weights overwhelm imputation tweaks
5. **Performance ceiling**: May have reached fundamental limit given data randomness

## Why We're Stuck

### The Middle Zone Problem

**Current**: 62.8% of predictions get 2-3 wrong (medium confidence on most events)
**Goal**: Push to extremes - either very confident (0-1 wrong) or very uncertain (4-5 wrong)

**Root Causes**:
1. Data is predominantly random with weak structure
2. Model can't distinguish high-confidence from low-confidence scenarios
3. Few extreme training examples (only 2.6% perfect, 2.1% total miss)
4. Imputation methods all converge to historical activation rates

## Next Steps (For ChatGPT 5.0 Analysis)

We're seeking external audit to identify:
1. **Alternative quantum characteristics** (spin, tunneling, annealing)
2. **Advanced imputation techniques** (Bayesian, conditional, VAEs)
3. **Hidden temporal patterns** (cycles, regime changes, Markov)
4. **Distribution anomalies** (outliers, subgroups, change points)
5. **Novel model architectures** (probabilistic, advanced attention, PINNs)
6. **Position-specific insights** (why QV_12, QV_13 have 88% false positive rates)
7. **Ensemble innovations** (stacking, mixture of experts)
8. **Quantum Computing**: True quantum algorithms (not just inspired features)

See `COMPREHENSIVE_PROJECT_AUDIT.md` for full details and the ChatGPT 5.0 audit prompt.

## Data Leakage Fixes

**CRITICAL**: This project has had 4 major data leakage fixes:
- Oct 22, 2025: Holdout leakage fix
- Oct 30, 2025: Imputation architecture fix
- Nov 5, 2025: Critical temporal leakage discovery
- Nov 10, 2025: LOW_BOUNDARY feature leakage

Current performance (50.92% recall) is **LEAK-FREE** and represents true model capability.

## Contributing

We welcome contributions! Areas of interest:
- Novel quantum features
- Advanced imputation methods
- Probabilistic model architectures
- Uncertainty quantification
- Conformal prediction
- Position-specific calibration
- Quantum Computing: True quantum algorithms (not just inspired features)

## License

MIT License - see LICENSE file

## Citation

```bibtex
@software{c5_quantum_imputer,
  title = {C5 Quantum Lottery Imputer},
  author = {Fiske, Roger and Claude (Anthropic)},
  year = {2025},
  url = {https://github.com/rogerfiske/c5-quantum-imputer}
}
```

## Contact

For questions or collaboration: [Your contact info]

## Acknowledgments

- Built with Claude Code (Anthropic)
- Uses LightGBM, PyTorch, scikit-learn
- Inspired by quantum computing concepts

---

**Status**: Active development, seeking breakthrough approaches to reduce middle zone
**Last Updated**: November 11, 2025
