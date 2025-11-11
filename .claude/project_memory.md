# C5 Quantum Imputer - Project Memory

## Project Overview
Quantum-inspired lottery prediction using imputation techniques to transform sparse binary data into rich quantum features for machine learning.

**Dataset**: 11,589 Cash 5 lottery events (5 winners from 39 positions)
**GitHub**: https://github.com/rogerfiske/c5-quantum-imputer
**Archive Project**: C:\Users\Minis\CascadeProjects\c5_new-idea (reference only, do not modify)

## Current Performance
- **Recall@20**: 50.92% (1000-event holdout, leak-free)
- **Random Baseline**: 25.6%
- **Data Structure**: 93% random, 7% exploitable patterns

## The Core Problem (Middle Zone)
**Current**: 62.8% of predictions get 2-3 wrong (stuck in middle zone)
**Goal**: Reduce to 35-40%, push predictions to extremes (0-1 wrong OR 4-5 wrong)

## What's Been Tried and Ruled Out

### Epic 13 (Nov 2025) - ALL FAILED
All approaches converged to identical 53.20% recall:
- Temporal weighting (multiple decay rates)
- Lookback window optimization (30-150 events)
- Ensemble voting mechanisms

### Epic 9B - Minimal Gain
Ensemble methods provided <0.3% improvement:
- Simple averaging
- Neural ensembles
- Temperature scaling

### All Imputation Methods Tested
- Amplitude Embedding (current best: 242 features)
- Density Matrix
- Angle Encoding
- Graph Cycle
All converge to similar probability estimates.

## Key Architectural Decisions

**Imputation Method**: Amplitude Embedding
- Transforms 5 binary winners → 242 quantum features
- Uses: amplitude, probability (Born rule), interference, entanglement, phase

**Model**: LightGBM Ranker
- Outperforms neural models (GNN, SetTransformer)
- Dominates imputation tweaks with learned weights

**Data Leakage**: Fixed 4 times (Oct-Nov 2025)
- Current performance is leak-free and reliable

## Known Patterns (7% Exploitable Structure)

1. **Position frequency bias**: QV_28 (14.2%) vs QV_1 (11.8%)
2. **Temporal persistence**: Recent positions 15% more likely
3. **Adjacent correlation**: QV_12 & QV_13 (+0.12 correlation)
4. **QV_12/QV_13 clustering**: 100% recall but 88% false positive rate (critical mystery)

## Current Status

**Awaiting**: ChatGPT 5.0 audit results
**Submitted**: COMPREHENSIVE_PROJECT_AUDIT.md with 10-area analysis request
**Goal**: Breakthrough recommendations beyond incremental tweaks

## When to Reference Archive

Archive location: `C:\Users\Minis\CascadeProjects\c5_new-idea`

**Contains**:
- Complete experimental history (Epic 1-13)
- All failed experimental scripts
- 100+ session summaries
- Diagnostic reports and model versions

**Use archive for**:
- Understanding "why didn't X work?"
- Historical performance data
- Reusing diagnostic/evaluation utilities

## Project Macro Objective

Transform wrong distribution from:
- Excellent/Good (0-1 wrong): 20.0% → Target 35-40%
- Poor (2-3 wrong): 62.8% → Target 35-40%
- Acceptable (4-5 wrong): 17.2% → OK

## Key Insight

Model can't distinguish high-confidence from low-confidence scenarios.
Need approaches that create prediction extremes, not medium-confidence on everything.

## Important Files

- `START_HERE_AFTER_CHATGPT_AUDIT.md` - Transition document for next session
- `COMPREHENSIVE_PROJECT_AUDIT.md` - Full 60+ page experimental history
- `DATASET_ANALYSIS.md` - Statistical analysis of 93% randomness
- `src/imputation/amplitude_embedding.py` - Primary imputation (242 features)
- `src/modeling/rankers/lgbm_ranker.py` - Best model architecture
- `production/models/amplitude/lgbm_ranker_baseline_fixed_v1.pkl` - Baseline model
