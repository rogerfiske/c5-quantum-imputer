# Epic 14: Extremizer Meta-Layer (Phase A)

**Status**: Draft
**Priority**: Highest (P0)
**Expected Impact**: -12 to -20 pp middle-zone reduction (62.8% → 42-50%), +0-2 pp Recall@20
**ChatGPT Audit Reference**: Phase A - Core extremizer implementation

## Epic Goal

Implement an extremizer meta-layer that reshapes event-level score distributions to reduce the "middle zone" problem (currently 62.8% of predictions get 2-3 wrong). The system will identify high-confidence vs low-confidence events and sharpen or flatten probability distributions accordingly, pushing predictions toward extremes (0-1 wrong OR 4-5 wrong).

## Root Cause Addressed

Event-level score vectors from the LGBM ranker are "moderately peaked" (neither sharp nor truly diffuse), causing predictions to cluster in the middle zone. Additionally, chronic false-positive positions (QV_12, QV_13, QV_4, QV_14, QV_22) inflate the tail of Top-20 selections, contributing to 2-3 hit outcomes.

## Success Metrics

**Primary Metric**: Middle-zone distribution (2-3 wrong)
- Current: 62.8%
- Target: 42-50% (conservative), 35-40% (stretch)

**Secondary Metric**: Recall@20
- Current: 50.92%
- Target: ≥50.92% (maintain or improve by +0-2 pp)

**Tertiary Metrics**:
- 0-1 wrong: increase from 20.0% to 25-30%
- 4-5 wrong: 17.2% (acceptable variance)

## Epic Architecture

```
Raw LGBM Scores (39 positions)
    ↓
[Position-Wise Isotonic Calibration]
    ↓
Calibrated Probabilities
    ↓
[Event-Level Diagnostics Engine]
    ↓
Diagnostic Features (entropy, Gini, top-K mass, gaps, HHI)
    ↓
[Middle-Zone Risk Classifier]
    ↓
Risk Score per Event
    ↓
[Temperature-Based Reshaping]
    ↓
Sharpened/Flattened Probabilities
    ↓
Top-20 Selection
```

## Stories

### Story 14.1: Position-Wise Isotonic Calibration Module

**As a** data scientist,
**I want** per-position isotonic calibration of raw LGBM scores,
**so that** chronic false-positive positions (QV_12/13/4/14/22) are deflated to calibrated probabilities.

**Acceptance Criteria**:
1. Implement `PositionIsotonicCalibrator` class with fit/transform/save/load methods
2. Calibrator trains 39 independent isotonic regressors (one per position)
3. Transform outputs renormalized probabilities (sum to 1 per event)
4. CLI supports fit mode (train + save) and apply mode (load + transform)
5. Includes utility functions: recall_at_20, distribution_breakdown, hits_at_k
6. All code has type hints, docstrings, and follows PEP8
7. Unit tests achieve ≥90% coverage
8. Integration test validates calibration reduces FP rate on QV_12/13

**File Deliverables**:
- `src/modeling/calibration/position_isotonic.py`
- `tests/modeling/calibration/test_position_isotonic.py`

---

### Story 14.2: Event-Level Diagnostics Engine

**As a** data scientist,
**I want** an event-level diagnostics engine that computes score shape features,
**so that** I can quantify whether an event's probability distribution is peaked or diffuse.

**Acceptance Criteria**:
1. Implement diagnostics computation module with functions for:
   - Shannon entropy
   - Gini coefficient
   - Top-K mass (K=1, 3, 5)
   - Herfindahl-Hirschman Index (HHI)
   - Top-5 score gaps (p[i] - p[i+1])
2. Diagnostics accept (n_events, 39) probability arrays
3. Output shape is (n_events, 11) feature matrix
4. Vectorized operations (no Python loops over events)
5. Numerical stability with epsilon clipping
6. Unit tests for each diagnostic function
7. Integration test validates diagnostics on synthetic peaked vs diffuse distributions

**File Deliverables**:
- `src/modeling/meta/diagnostics.py`
- `tests/modeling/meta/test_diagnostics.py`

---

### Story 14.3: Middle-Zone Risk Classifier

**As a** data scientist,
**I want** a trained classifier that predicts middle-zone risk per event,
**so that** the extremizer can decide whether to sharpen or flatten probability distributions.

**Acceptance Criteria**:
1. Implement `MiddleZoneRiskClassifier` using LogisticRegression
2. Target variable: binary indicator if event lands in {2,3} hits within Top-20
3. Features: 11 diagnostics from Story 14.2
4. Training supports GroupKFold CV with AUC reporting
5. Outputs calibrated risk probabilities [0, 1]
6. Includes fit/predict_proba/save/load methods
7. Unit tests validate classifier training and prediction
8. Integration test achieves AUC ≥ 0.60 on synthetic data

**File Deliverables**:
- `src/modeling/meta/risk_classifier.py`
- `tests/modeling/meta/test_risk_classifier.py`

---

### Story 14.4: Temperature-Based Sharpening/Flattening

**As a** data scientist,
**I want** a temperature-based probability reshaping function,
**so that** low-risk events are sharpened and high-risk events are flattened.

**Acceptance Criteria**:
1. Implement `apply_gamma` function: p_new = p^gamma / sum(p^gamma)
2. Configurable gamma_sharp (default: 1.6) for low-risk events
3. Configurable gamma_flat (default: 0.8) for high-risk events
4. Risk threshold parameter (default: 0.5) maps risk → gamma
5. Vectorized batch processing of event arrays
6. Numerical stability with clipping [1e-12, 1.0]
7. Unit tests validate sharpening increases top-1 mass
8. Unit tests validate flattening increases entropy

**File Deliverables**:
- `src/modeling/meta/temperature_shaping.py`
- `tests/modeling/meta/test_temperature_shaping.py`

---

### Story 14.5: Extremizer Pipeline Integration & Evaluation

**As a** data scientist,
**I want** the complete extremizer pipeline integrated with existing LGBM baseline,
**so that** I can evaluate impact on middle-zone distribution and Recall@20.

**Acceptance Criteria**:
1. Implement `Extremizer` class orchestrating Stories 14.1-14.4
2. Fit method trains calibrator + diagnostics + risk classifier
3. Transform method applies full pipeline: calibrate → diagnose → classify → reshape → Top-20
4. CLI interface for training and prediction
5. Evaluation script measures:
   - Recall@20 (before vs after)
   - Distribution breakdown (0-1 / 2-3 / 4-5 wrong)
   - Per-position FP rate changes
6. Grid search over (risk_threshold, gamma_sharp, gamma_flat)
7. Integration test on 1000-event holdout validates:
   - Middle-zone reduction ≥ 8 pp
   - Recall@20 degradation ≤ 1 pp
8. Results logged to `.ai/debug-log.md`

**File Deliverables**:
- `src/modeling/meta/extremizer.py`
- `tests/modeling/meta/test_extremizer.py`
- `scripts/evaluate_extremizer.py`
- `.ai/epic-14-evaluation-results.md`

---

## Dependencies

**Required Files** (from ChatGPT audit):
- `ChatGPT_response.txt` - extremizer_meta.py code snippet
- Existing LGBM baseline: `production/models/amplitude/lgbm_ranker_baseline_fixed_v1.pkl`
- Holdout test dataset: used by `production/comprehensive_1000event_holdout_diagnostic.py`

**Architecture Documents Needed** (Story 14.1 will create):
- `docs/architecture/testing-strategy.md` - pytest standards
- `docs/architecture/coding-standards.md` - Python style guide
- `docs/architecture/tech-stack.md` - dependency versions

## Technical Constraints

- **Python**: 3.8+
- **Dependencies**: numpy, pandas, scikit-learn, joblib, pyarrow
- **Testing**: pytest, pytest-cov (≥90% coverage)
- **Validation**: Strict rolling/blocked CV (no temporal leakage)
- **Performance**: Calibration + reshaping must execute <5s for 1000 events

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Calibration overfits to training distribution | Medium | Use GroupKFold CV, validate on multiple holdout folds |
| Risk classifier AUC too low (<0.55) | High | Add polynomial features, try XGBoost if logistic fails |
| Gamma parameters too aggressive | Medium | Grid search with Recall@20 constraint (≥50.5%) |
| Integration breaks existing pipeline | Low | Wrapper pattern, existing baseline unchanged |

## Definition of Done

- [ ] All 5 stories completed with Status: Done
- [ ] All acceptance criteria met and validated
- [ ] Test coverage ≥90% for all new modules
- [ ] Integration test on 1000-event holdout shows:
  - Middle-zone: 42-54% (target met if ≤54%)
  - Recall@20: ≥50.5%
- [ ] Evaluation results documented in `.ai/epic-14-evaluation-results.md`
- [ ] Code committed to git with clear commit messages
- [ ] Ready for Epic 15 (Ising re-ranker integration)

---

**Created**: 2025-11-11
**Author**: James (Dev Agent)
**ChatGPT Audit Phase**: A (Core)
