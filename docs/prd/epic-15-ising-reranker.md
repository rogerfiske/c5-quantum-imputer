# Epic 15: Ising Interaction Re-Ranker (Phase B)

**Status**: Draft
**Priority**: High (P1)
**Expected Impact**: -2 to -5 pp additional middle-zone reduction, +0.5-1.0 pp Recall@20
**ChatGPT Audit Reference**: Phase B - Additive interaction layer
**Depends On**: Epic 14 (Extremizer Meta-Layer)

## Epic Goal

Implement a lightweight Ising-inspired pairwise interaction re-ranker that learns symmetric couplings between positions from historical co-occurrence patterns. This adds an interaction-aware synergy term to calibrated probabilities, explicitly rewarding/penalizing position co-activations to push predictions away from ambiguous middle-zone outcomes.

## Root Cause Addressed

Current LGBM ranker treats positions independently. Real lottery data exhibits weak but exploitable pairwise dependencies (e.g., QV_12 & QV_13 co-occur with +0.12 correlation). By modeling these interactions explicitly, we can re-rank candidates to prefer decisive cliques (sharpen) or avoid ambiguous co-sets (flatten), further reducing middle-zone mass.

## Success Metrics

**Primary Metric**: Middle-zone distribution (2-3 wrong)
- Post-Epic 14: ~42-50%
- Post-Epic 15 Target: 38-47% (-2 to -5 pp)

**Secondary Metric**: Recall@20
- Post-Epic 14: ~51-53%
- Post-Epic 15 Target: 51.5-54% (+0.5-1.0 pp)

**Tertiary Metrics**:
- 0-1 wrong: increase by +1-2 pp
- Pairwise synergy AUC: ≥0.55 (interaction signal detection)

## Epic Architecture

```
Calibrated Probabilities (from Epic 14)
    ↓
[Empirical Co-occurrence Statistics]
    ↓
Unary Fields (h) + Pairwise Couplings (J)
    ↓
[Synergy Augmentation]
Augmented Scores = p + λ * (p @ J.T)
    ↓
Renormalized Probabilities
    ↓
Top-20 Re-Ranking
```

## Core Innovation

**Lightweight Ising Model**:
- Unary fields: `h_i = logit(P(x_i=1))`
- Pairwise couplings: `J_ij ~ log(P(i,j) / (P(i)*P(j)))`
- Energy: `E(x) = -Σh_i*x_i - ΣJ_ij*x_i*x_j`
- Augmentation: Add synergy term λ*synergy to base scores

**Advantages**:
- No heavy optimization (closed-form from counts)
- Stable, interpretable parameters
- Drop-in augmentation on top of calibrated probabilities

## Stories

### Story 15.1: Empirical Co-occurrence Statistics Engine

**As a** data scientist,
**I want** a module that computes empirical unary and pairwise statistics from binary labels,
**so that** I can estimate position activation rates and co-activation frequencies.

**Acceptance Criteria**:
1. Implement `empirical_stats(Y, eps)` function
2. Input: binary label matrix (n_events, 39)
3. Output: (p1, p2) where:
   - p1[i] = smoothed P(position_i = 1)
   - p2[i,j] = smoothed P(position_i=1, position_j=1) (symmetric, zero diagonal)
4. Laplace smoothing with configurable epsilon (default: 1e-6)
5. Vectorized computation using matrix multiplication
6. Unit tests validate p1 sums reasonably, p2 is symmetric
7. Integration test on synthetic data with known correlations

**File Deliverables**:
- `src/modeling/ensembles/ising_stats.py`
- `tests/modeling/ensembles/test_ising_stats.py`

---

### Story 15.2: Ising Parameter Estimation

**As a** data scientist,
**I want** a module that converts empirical statistics to Ising fields and couplings,
**so that** I can represent position interactions as log-odds ratios.

**Acceptance Criteria**:
1. Implement `estimate_ising_parameters(p1, p2, clip_j)` function
2. Unary fields: `h_i = logit(p1[i])` with safe clipping
3. Pairwise couplings: `J_ij = log(p2[i,j] / (p1[i]*p1[j]))` with clipping
4. J matrix is symmetric, zero diagonal
5. J magnitude clipped to [-clip_j, +clip_j] (default: 2.0)
6. Optional centering: `J = J - row_mean - col_mean + global_mean`
7. Unit tests validate J symmetry, diagonal zeros, magnitude bounds
8. Integration test on synthetic data with planted interactions

**File Deliverables**:
- `src/modeling/ensembles/ising_parameters.py`
- `tests/modeling/ensembles/test_ising_parameters.py`

---

### Story 15.3: Synergy Augmentation Layer

**As a** data scientist,
**I want** a synergy augmentation function that adds interaction terms to base scores,
**so that** positions with strong couplings to already-probable positions get boosted.

**Acceptance Criteria**:
1. Implement `augment_scores(p, J, lambda_pair)` function
2. Synergy computation: `synergy = p @ J.T` (expected neighbor activation)
3. Augmented scores: `s_aug = p + lambda_pair * synergy`
4. Renormalization: `s_aug = s_aug / sum(s_aug)` per event
5. Configurable lambda_pair (default: 0.4)
6. Vectorized batch processing
7. Unit tests validate synergy shape, renormalization
8. Integration test shows high-coupling positions receive boosts

**File Deliverables**:
- `src/modeling/ensembles/synergy_augmentation.py`
- `tests/modeling/ensembles/test_synergy_augmentation.py`

---

### Story 15.4: Lightweight Ising Re-Ranker Integration

**As a** data scientist,
**I want** a complete `LightweightIsingReRanker` class,
**so that** I can train on historical labels and re-rank calibrated probabilities.

**Acceptance Criteria**:
1. Implement `LightweightIsingReRanker` class with fit/augment_scores/topk_indices methods
2. Fit method:
   - Computes empirical stats from labels
   - Estimates h and J parameters
   - Stores for inference
3. Augment_scores method applies synergy augmentation
4. CLI interface for fit_apply mode
5. Includes save/load via joblib
6. Unit tests for fit/transform workflow
7. Integration test validates re-ranking changes Top-20 selections
8. Performance: <2s for 1000 events

**File Deliverables**:
- `src/modeling/ensembles/ising_reranker.py` (consolidates Stories 15.1-15.3)
- `tests/modeling/ensembles/test_ising_reranker.py`

---

### Story 15.5: Ising Re-Ranker Evaluation & Hyperparameter Tuning

**As a** data scientist,
**I want** an evaluation pipeline for the Ising re-ranker,
**so that** I can measure impact on middle-zone distribution and tune lambda_pair.

**Acceptance Criteria**:
1. Evaluation script integrates Ising re-ranker with Epic 14 extremizer
2. Pipeline: LGBM → Extremizer → Ising Re-Ranker → Top-20
3. Metrics tracked:
   - Recall@20 (baseline, extremizer-only, extremizer+Ising)
   - Distribution breakdown (0-1 / 2-3 / 4-5 wrong)
   - Pairwise synergy AUC (J predicts co-occurrence)
4. Grid search over lambda_pair ∈ [0.0, 0.2, 0.4, 0.6, 0.8]
5. Grid search over clip_j ∈ [1.0, 2.0, 3.0]
6. Integration test on 1000-event holdout validates:
   - Middle-zone reduction ≥ 2 pp vs extremizer-only
   - Recall@20 improvement ≥ 0.3 pp
7. Results logged to `.ai/epic-15-evaluation-results.md`

**File Deliverables**:
- `scripts/evaluate_ising_reranker.py`
- `tests/integration/test_epic15_pipeline.py`
- `.ai/epic-15-evaluation-results.md`

---

## Dependencies

**Required Files**:
- Epic 14 output: `src/modeling/meta/extremizer.py`
- ChatGPT code: `ChatGPT_response.txt` (ising_reranker.py snippet)
- Holdout data: from `production/comprehensive_1000event_holdout_diagnostic.py`

**Python Dependencies**:
- numpy, pandas, scikit-learn, joblib, pyarrow
- pytest, pytest-cov

## Technical Constraints

- **Coupling sparsity**: J matrix is (39, 39) but we enforce symmetry + zero diagonal
- **Stability**: All log operations must clip inputs to avoid inf/nan
- **Performance**: Full pipeline (calibration + extremizer + Ising) must run <10s for 1000 events
- **Validation**: Rolling CV, no temporal leakage

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Empirical J is too noisy (93% random data) | High | Heavy clipping (clip_j=2.0), centering, smoothing |
| Lambda too aggressive → overfits co-occurrence | Medium | Conservative grid search, validate on multiple folds |
| Synergy term has minimal signal | Low | If AUC <0.52, skip Epic 15 and proceed to Epic 16 |
| Integration slows pipeline | Low | Profile code, use vectorized ops |

## Definition of Done

- [ ] All 5 stories completed with Status: Done
- [ ] All acceptance criteria met and validated
- [ ] Test coverage ≥90% for all new modules
- [ ] Integration test on 1000-event holdout shows:
  - Middle-zone: ≤45% (improvement over extremizer-only)
  - Recall@20: ≥51.5%
  - Pairwise synergy AUC ≥0.53
- [ ] Evaluation results documented in `.ai/epic-15-evaluation-results.md`
- [ ] Code committed to git with clear commit messages
- [ ] Ready for Epic 16 (Mixture-of-Experts)

---

**Created**: 2025-11-11
**Author**: James (Dev Agent)
**ChatGPT Audit Phase**: B (Additive)
