# Epic 16: Mixture-of-Experts Ensemble (Phase C)

**Status**: Draft
**Priority**: Medium (P2)
**Expected Impact**: -2 to -4 pp additional middle-zone reduction, +0-1 pp Recall@20
**ChatGPT Audit Reference**: Phase C - Gated experts
**Depends On**: Epic 14 (Extremizer), Epic 15 (Ising Re-Ranker)

## Epic Goal

Implement a Mixture-of-Experts (MoE) ensemble that intelligently routes events to specialized prediction policies based on event-level diagnostics. The system will gate between three experts (Sharpened+Calibrated, Flattened+Calibrated, Ising-Reranked) using the middle-zone risk classifier, with optional stacking near decision boundaries.

## Root Cause Addressed

Not all events benefit equally from sharpening, flattening, or interaction modeling. Some events have decisive patterns that justify aggressive sharpening; others are irreducibly noisy and should be flattened; still others exhibit strong pairwise structure suited for Ising re-ranking. A gated MoE can select the optimal policy per event, further reducing middle-zone mass beyond single-policy approaches.

## Success Metrics

**Primary Metric**: Middle-zone distribution (2-3 wrong)
- Post-Epic 15: ~38-47%
- Post-Epic 16 Target: 35-43% (-2 to -4 pp)

**Secondary Metric**: Recall@20
- Post-Epic 15: ~51.5-54%
- Post-Epic 16 Target: 51.5-55% (neutral to +1 pp)

**Tertiary Metrics**:
- Expert routing accuracy: ≥60% (correct expert chosen)
- Diversity coefficient: ≥0.3 (experts produce different Top-20s)

## Epic Architecture

```
Calibrated Probabilities + Diagnostics
    ↓
[Gating Network]
    ↓
Gate Probabilities → [Expert A: Sharpened]
                  → [Expert B: Flattened]
                  → [Expert C: Ising-Reranked]
    ↓
[Expert Outputs]
    ↓
[Fusion Strategy]
    ↓
Final Top-20 Selection
```

## Gating Strategy

**Hard Gating** (Phase C.1):
- If risk < threshold_low → Expert A (Sharpened)
- If risk > threshold_high → Expert B (Flattened)
- Else → Expert C (Ising-Reranked)

**Soft Gating** (Phase C.2 - optional):
- Weighted combination of expert outputs
- Weights from calibrated gate probabilities

**Stacking Meta-Learner** (Phase C.3 - optional):
- Near decision boundaries (threshold ± margin), blend expert outputs via logistic stacker

## Stories

### Story 16.1: Expert Routing & Gating Mechanism

**As a** data scientist,
**I want** a gating network that routes events to specialized experts,
**so that** each event is processed by the most suitable prediction policy.

**Acceptance Criteria**:
1. Implement `ExpertGate` class with routing logic
2. Input: event-level diagnostics + risk scores (from Epic 14)
3. Output: expert assignment per event (A/B/C)
4. Configurable thresholds: risk_low (default: 0.35), risk_high (default: 0.65)
5. Supports hard gating (winner-take-all)
6. Supports soft gating (weighted combination)
7. Unit tests validate routing logic with synthetic risk scores
8. Integration test shows balanced expert utilization (no single expert >70%)

**File Deliverables**:
- `src/modeling/ensembles/expert_gate.py`
- `tests/modeling/ensembles/test_expert_gate.py`

---

### Story 16.2: Expert Policy Implementations

**As a** data scientist,
**I want** three distinct expert policies encapsulated as callable modules,
**so that** the MoE can route events to specialized prediction strategies.

**Acceptance Criteria**:
1. Implement `ExpertA` (Sharpened + Calibrated):
   - Uses gamma_sharp = 1.8 (aggressive sharpening)
   - Applies position-wise calibration from Epic 14
2. Implement `ExpertB` (Flattened + Calibrated):
   - Uses gamma_flat = 0.7 (aggressive flattening)
   - Applies position-wise calibration from Epic 14
3. Implement `ExpertC` (Ising-Reranked):
   - Uses Ising re-ranker from Epic 15 with lambda_pair = 0.5
4. All experts accept (n_events, 39) probabilities → return (n_events, 39) adjusted probabilities
5. Unit tests for each expert validate output shape and properties
6. Integration test validates experts produce different Top-20 rankings

**File Deliverables**:
- `src/modeling/ensembles/expert_policies.py`
- `tests/modeling/ensembles/test_expert_policies.py`

---

### Story 16.3: Diversity Regularization & Measurement

**As a** data scientist,
**I want** diversity metrics and optional regularization,
**so that** experts don't converge to identical predictions (avoiding redundancy).

**Acceptance Criteria**:
1. Implement `diversity_coefficient(top20_A, top20_B)` function
   - Measures Jaccard distance between Top-20 sets
   - Returns value ∈ [0, 1] (0=identical, 1=completely different)
2. Implement `expert_diversity_matrix(expert_outputs)` for all pairs
3. Optional diversity regularization term in stacking (Story 16.4)
4. Unit tests validate diversity computation
5. Integration test measures diversity across experts on holdout data
6. Target: pairwise diversity ≥0.25 for all expert pairs

**File Deliverables**:
- `src/modeling/ensembles/diversity_metrics.py`
- `tests/modeling/ensembles/test_diversity_metrics.py`

---

### Story 16.4: Stacking Meta-Learner (Optional Soft Fusion)

**As a** data scientist,
**I want** a stacking meta-learner that blends expert outputs near gating boundaries,
**so that** events with ambiguous routing benefit from multiple expert signals.

**Acceptance Criteria**:
1. Implement `StackingMetaLearner` using LogisticRegression
2. Features: concatenated expert Top-K scores (K=30) + diagnostics
3. Target: binary indicator per position (top-20 inclusion)
4. Training uses GroupKFold CV
5. Blending triggered when risk ∈ [threshold_low - margin, threshold_high + margin]
6. Configurable margin (default: 0.15)
7. Unit tests validate stacker training and prediction
8. Integration test shows blending improves recall on boundary events ≥0.5 pp

**File Deliverables**:
- `src/modeling/ensembles/stacking_meta.py`
- `tests/modeling/ensembles/test_stacking_meta.py`

---

### Story 16.5: MoE Pipeline Integration & Evaluation

**As a** data scientist,
**I want** the complete MoE pipeline integrated and evaluated,
**so that** I can measure final impact on middle-zone distribution and Recall@20.

**Acceptance Criteria**:
1. Implement `MixtureOfExperts` orchestrator class
2. Components: ExpertGate + 3 Expert Policies + optional Stacking
3. Fit method trains all components on historical data
4. Transform method routes events and applies selected experts
5. CLI interface for training and prediction
6. Evaluation script measures:
   - Recall@20 (baseline, Epic 14, Epic 15, Epic 16)
   - Distribution breakdown at each stage
   - Expert utilization percentages
   - Diversity metrics
7. Grid search over (risk_low, risk_high, stacking_margin)
8. Integration test on 1000-event holdout validates:
   - Middle-zone: ≤43%
   - Recall@20: ≥52%
   - Expert diversity: ≥0.25
9. Results logged to `.ai/epic-16-evaluation-results.md`

**File Deliverables**:
- `src/modeling/ensembles/mixture_of_experts.py`
- `tests/modeling/ensembles/test_mixture_of_experts.py`
- `scripts/evaluate_moe.py`
- `.ai/epic-16-evaluation-results.md`

---

## Dependencies

**Required Files**:
- Epic 14: `src/modeling/meta/extremizer.py`
- Epic 15: `src/modeling/ensembles/ising_reranker.py`
- Holdout data: from `production/comprehensive_1000event_holdout_diagnostic.py`

**Python Dependencies**:
- numpy, pandas, scikit-learn, joblib, pyarrow
- pytest, pytest-cov

## Technical Constraints

- **Gating overhead**: Routing decision must execute <0.1s per 1000 events
- **Expert parallelization**: Experts can run in parallel for soft gating (future optimization)
- **Stacking complexity**: Meta-learner limited to logistic regression (avoid overfitting)
- **Validation**: Rolling CV, no temporal leakage

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Experts converge to similar outputs (low diversity) | High | Enforce different gamma values, measure diversity |
| Gating thresholds are dataset-specific | Medium | Grid search with validation, document sensitivity |
| Stacking overfits to training fold | Medium | Strict GroupKFold, limit features to top-K scores |
| Marginal gains (<2 pp) don't justify complexity | Low | Make stacking optional, evaluate hard vs soft gating |

## Definition of Done

- [ ] All 5 stories completed with Status: Done
- [ ] All acceptance criteria met and validated
- [ ] Test coverage ≥90% for all new modules
- [ ] Integration test on 1000-event holdout shows:
  - Middle-zone: ≤43%
  - Recall@20: ≥52%
  - Expert diversity: ≥0.25 pairwise
- [ ] Evaluation results documented in `.ai/epic-16-evaluation-results.md`
- [ ] Code committed to git with clear commit messages
- [ ] Ready for Epic 17 (Governance & Observability)

---

**Created**: 2025-11-11
**Author**: James (Dev Agent)
**ChatGPT Audit Phase**: C (Experts)
