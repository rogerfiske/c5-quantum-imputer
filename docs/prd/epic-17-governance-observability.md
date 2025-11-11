# Epic 17: Governance & Observability (Phase D)

**Status**: Draft
**Priority**: Medium (P2)
**Expected Impact**: Operational stability, auditability, and drift detection
**ChatGPT Audit Reference**: Phase D - Governance layer
**Depends On**: Epic 14-16 (complete prediction pipeline)

## Epic Goal

Implement governance and observability infrastructure to ensure the prediction pipeline remains stable, auditable, and adaptive over time. This includes change-point detection for recalibration triggers, interpretable subgroup rules for decision transparency, calibration drift monitoring, and deployment checklists.

## Root Cause Addressed

Machine learning systems degrade over time due to:
- **Distribution drift**: Lottery drawing patterns may shift subtly
- **Calibration decay**: Position-wise FP rates may change
- **Regime changes**: Temporal patterns identified by Epic 13 may emerge/disappear

Without observability and governance, the pipeline will:
- Silently degrade in performance
- Lack auditability (black-box decisions)
- Require manual monitoring and ad-hoc recalibration

## Success Metrics

**Primary Metrics**:
- Change-point detection latency: <10 events after regime shift
- Recalibration trigger accuracy: ≥70% (true positives)

**Secondary Metrics**:
- Subgroup rule coverage: ≥60% of events match ≥1 rule
- Subgroup rule precision: ≥65% (when rule fires → correct sharpening/flattening)
- Calibration drift alert rate: <5% false positives

**Operational Metrics**:
- Monitoring dashboard latency: <5s for 1000-event batch
- Deployment checklist completion time: <30 min

## Epic Architecture

```
Production Pipeline
    ↓
[Change-Point Detection (BOCPD)]
    ↓
Alert if P(change) > threshold
    ↓
[Calibration Drift Monitor]
    ↓
Track per-position FP rates, ECE
    ↓
[Subgroup Rule Evaluator]
    ↓
Match event diagnostics to interpretable rules
    ↓
[Logging & Dashboard]
    ↓
Store metrics, visualize trends
```

## Stories

### Story 17.1: Bayesian Online Change-Point Detection (BOCPD)

**As a** ML engineer,
**I want** online change-point detection on position frequencies,
**so that** I can trigger recalibration when drawing patterns shift.

**Acceptance Criteria**:
1. Implement `BayesianChangePointDetector` using BOCPD algorithm
2. Input: stream of events with binary position labels
3. Track per-position activation rate over sliding window (default: 50 events)
4. Compute run-length posterior P(r_t) and change probability P(change_t)
5. Alert when P(change) > threshold (default: 0.3)
6. Configurable hazard rate (default: 1/100)
7. Unit tests validate change detection on synthetic step-change data
8. Integration test detects change within 10 events of true change-point

**File Deliverables**:
- `src/monitoring/change_point_detection.py`
- `tests/monitoring/test_change_point_detection.py`

---

### Story 17.2: Subgroup Rule Discovery with SkopeRules

**As a** ML engineer,
**I want** interpretable subgroup rules that identify high-confidence vs low-confidence events,
**so that** the system's decisions are auditable and explainable.

**Acceptance Criteria**:
1. Implement `SubgroupRuleDiscovery` using SkopeRules or L1-logistic
2. Input: event diagnostics (entropy, Gini, etc.) + binary target (should_sharpen)
3. Output: 5-15 simple rules (e.g., "IF entropy < 2.5 AND top1_gap > 0.05 THEN sharpen")
4. Rules optimized for precision ≥65%
5. Rule coverage: ≥60% of events match at least one rule
6. CLI interface to train rules and export to JSON
7. Unit tests validate rule extraction and evaluation
8. Integration test on holdout data achieves target precision and coverage

**File Deliverables**:
- `src/monitoring/subgroup_rules.py`
- `tests/monitoring/test_subgroup_rules.py`
- `.ai/subgroup-rules.json` (exported rules)

---

### Story 17.3: Calibration Drift Monitoring

**As a** ML engineer,
**I want** calibration drift monitoring that tracks per-position FP rates and ECE,
**so that** I can detect when the calibrator needs retraining.

**Acceptance Criteria**:
1. Implement `CalibrationDriftMonitor` class
2. Track metrics per position:
   - False Positive rate (actual vs expected)
   - Expected Calibration Error (ECE) via reliability diagrams
3. Sliding window computation (default: 100 events)
4. Alert if:
   - Any position FP rate deviates >10% from training baseline
   - Global ECE increases >5% from training baseline
5. Configurable alert thresholds
6. Unit tests validate ECE computation
7. Integration test detects drift on synthetic mis-calibrated data

**File Deliverables**:
- `src/monitoring/calibration_drift.py`
- `tests/monitoring/test_calibration_drift.py`

---

### Story 17.4: Monitoring Dashboard & Logging Infrastructure

**As a** ML engineer,
**I want** a monitoring dashboard that visualizes pipeline health metrics,
**so that** I can track performance trends and detect issues early.

**Acceptance Criteria**:
1. Implement logging infrastructure:
   - Event-level predictions logged to `.ai/predictions-log.jsonl`
   - Metrics aggregated to `.ai/metrics-summary.json` (daily)
2. Dashboard script generates:
   - Recall@20 trend over time
   - Middle-zone distribution trend
   - Per-position FP rate heatmap
   - Change-point alerts timeline
   - Calibration drift alerts
3. Dashboard outputs static HTML (no server required)
4. Refresh latency: <5s for 1000-event batch
5. Unit tests validate logging format
6. Integration test generates dashboard on synthetic event stream

**File Deliverables**:
- `src/monitoring/logger.py`
- `src/monitoring/dashboard.py`
- `tests/monitoring/test_logger.py`
- `tests/monitoring/test_dashboard.py`
- `scripts/generate_dashboard.py`

---

### Story 17.5: Deployment Checklist & Recalibration Workflow

**As a** ML engineer,
**I want** a deployment checklist and automated recalibration workflow,
**so that** model updates are safe, reproducible, and auditable.

**Acceptance Criteria**:
1. Create `.bmad-core/checklists/model-deployment-checklist.md`
2. Checklist includes:
   - [ ] Holdout test results (Recall@20, middle-zone)
   - [ ] Calibration metrics (ECE, FP rates)
   - [ ] Change-point detection status
   - [ ] Subgroup rule validation
   - [ ] Code review + test coverage ≥90%
   - [ ] Commit SHA and deployment timestamp
3. Implement `recalibration_workflow.py` script:
   - Detects change-point alert
   - Retains last N events (default: 200) as new calibration set
   - Refits calibrator (preserves old version as backup)
   - Validates new calibrator on rolling window
   - Logs recalibration event
4. CLI interface for manual recalibration trigger
5. Unit tests validate recalibration logic
6. Integration test simulates change-point → recalibration → validation

**File Deliverables**:
- `.bmad-core/checklists/model-deployment-checklist.md`
- `src/monitoring/recalibration_workflow.py`
- `tests/monitoring/test_recalibration_workflow.py`
- `scripts/recalibrate_model.py`

---

## Dependencies

**Required Files**:
- Epic 14-16: Complete prediction pipeline
- Historical event logs (for BOCPD and drift monitoring)
- Deployment environment configuration

**Python Dependencies**:
- numpy, pandas, scikit-learn, joblib
- skope-rules (for Story 17.2)
- matplotlib, plotly (for dashboard)
- pytest, pytest-cov

## Technical Constraints

- **Monitoring overhead**: <5% latency increase on prediction pipeline
- **Storage**: Event logs capped at 10,000 most recent events (rolling buffer)
- **Dashboard refresh**: Static HTML generation (no live server for simplicity)
- **Recalibration**: Automated only for drift alerts; manual approval required for production deployment

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| BOCPD false positives trigger unnecessary recalibration | Medium | Conservative threshold (0.3), require 3 consecutive alerts |
| Subgroup rules overfit to training data | Medium | Validate on multiple folds, require precision ≥65% |
| Dashboard generation slows pipeline | Low | Async generation, cache metrics |
| Recalibration introduces regressions | High | Always validate new calibrator on holdout before deployment |

## Definition of Done

- [ ] All 5 stories completed with Status: Done
- [ ] All acceptance criteria met and validated
- [ ] Test coverage ≥90% for all new modules
- [ ] Integration test validates full governance pipeline:
  - BOCPD detects synthetic change-point within 10 events
  - Subgroup rules achieve ≥65% precision, ≥60% coverage
  - Calibration drift monitor flags mis-calibrated positions
  - Dashboard generates in <5s
  - Recalibration workflow completes without errors
- [ ] Deployment checklist validated on Epic 14-16 pipeline
- [ ] Documentation in `.ai/governance-documentation.md`
- [ ] Code committed to git with clear commit messages

---

**Created**: 2025-11-11
**Author**: James (Dev Agent)
**ChatGPT Audit Phase**: D (Governance)
