# Comprehensive 1000-Event Holdout Diagnostic Report

**Test Date**: 2025-11-11 10:23:30
**Model**: lgbm_ranker_baseline_fixed_v1.pkl (Fixed Architecture - NO LEAKAGE)
**Events Tested**: 1000

## HOLDOUT TEST SUMMARY - 1000 Events
```
  ------------------------------------
  0 wrong:   26 events ( 2.60%)  ← All 5 actual values in top-20  ✓ EXCELLENT
  1 wrong:  174 events (17.40%)  ← 4 of 5 actual values in top-20  ✓ GOOD
  2 wrong:  313 events (31.30%)  ← 3 of 5 actual values in top-20  ✗ POOR
  3 wrong:  315 events (31.50%)  ← 2 of 5 actual values in top-20  ✗ POOR
  4 wrong:  151 events (15.10%)  ← 1 of 5 actual values in top-20  ~ ACCEPTABLE
  5 wrong:   21 events ( 2.10%)  ← 0 of 5 actual values in top-20  ~ ACCEPTABLE
  ------------------------------------

Overall Recall@20: 50.92% (2546/5000 correct)
```

## MACRO OBJECTIVE ANALYSIS

**Goal**: Force results to EXTREMES (0-1 wrong OR 4-5 wrong)

- **Excellent/Good (0-1 wrong)**: 20.0% ← WANT HIGHER
- **Poor (2-3 wrong)**: 62.8% ← WANT LOWER (stuck in middle!)
- **Acceptable (4-5 wrong)**: 17.2% ← OK

## TOP IMPROVEMENT TARGETS

### Positions Stuck in Medium Confidence (0.4-0.6)
These positions need stronger signals to push to extremes:


### High False Positive Positions
These are predicted too often when wrong:

- **QV 12**: FP rate=88.9%, recall=100.0%
- **QV 13**: FP rate=88.7%, recall=100.0%
- **QV 4**: FP rate=88.6%, recall=99.1%
- **QV 14**: FP rate=88.5%, recall=100.0%
- **QV 22**: FP rate=88.4%, recall=100.0%
- **QV 1**: FP rate=87.9%, recall=100.0%
- **QV 5**: FP rate=87.9%, recall=100.0%
- **QV 8**: FP rate=87.7%, recall=100.0%
- **QV 6**: FP rate=87.5%, recall=100.0%
- **QV 3**: FP rate=87.3%, recall=100.0%

## ACTIONABLE RECOMMENDATIONS

### Algorithm Modifications

#### Ensemble Confidence Voting [MEDIUM]

**Problem**: Single model produces middle-ground predictions

**Solution**: Use multiple imputation methods, vote only on high-agreement positions

**Expected Impact**: Force more predictions to extremes (high confidence = agree, low = disagree)

**Implementation**: Create ensemble of: frequency, pattern-based, temporal-weighted

### Imputation Improvements

#### Temporal Weighting in Imputation [MEDIUM]

**Problem**: Simple frequency treats all 50 events equally

**Solution**: Weight recent events more heavily (exponential decay)

**Expected Impact**: Better capture trends, improve recall by 5-10%

**Implementation**: Modify _impute_next_event_probabilities() to apply decay weights

