# Epic 14-17 Roadmap - ChatGPT Audit Implementation

**Created**: 2025-11-11
**Status**: Ready for Story 14.1 Implementation
**Based On**: ChatGPT 5.0 Audit Response (ChatGPT_response.txt)

## Summary

Successfully created a complete 4-epic roadmap to address the middle-zone problem (62.8% of predictions get 2-3 wrong) based on ChatGPT's comprehensive audit recommendations. All epic definitions are complete, and 3 production-ready modules have been integrated into the codebase.

## Epic Structure

### Epic 14: Extremizer Meta-Layer (Phase A) - **HIGHEST PRIORITY**
**Impact**: -12 to -20 pp middle-zone reduction
**File**: `docs/prd/epic-14-extremizer-meta-layer.md`
**Stories**: 14.1 through 14.5
**Focus**: Position-wise calibration + event-level diagnostics + risk classification + temperature reshaping

### Epic 15: Ising Interaction Re-Ranker (Phase B)
**Impact**: -2 to -5 pp additional middle-zone reduction
**File**: `docs/prd/epic-15-ising-reranker.md`
**Stories**: 15.1 through 15.5
**Focus**: Pairwise co-occurrence modeling + synergy augmentation

### Epic 16: Mixture-of-Experts (Phase C)
**Impact**: -2 to -4 pp additional middle-zone reduction
**File**: `docs/prd/epic-16-mixture-of-experts.md`
**Stories**: 16.1 through 16.5
**Focus**: Gated expert routing + diversity regularization + optional stacking

### Epic 17: Governance & Observability (Phase D)
**Impact**: Operational stability and drift detection
**File**: `docs/prd/epic-17-governance-observability.md`
**Stories**: 17.1 through 17.5
**Focus**: Change-point detection + calibration drift monitoring + recalibration workflows

## Integrated Code Modules (from ChatGPT)

### 1. Position-Wise Isotonic Calibration
**File**: `src/modeling/calibration/position_isotonic.py`
**Purpose**: Per-position calibration to reduce chronic FP positions (QV_12/13/4/14/22)
**Status**: ✅ Integrated
**Key Classes**: `PositionIsotonicCalibrator`, `CalibrationPackConfig`

### 2. Ising Re-Ranker
**File**: `src/modeling/ensembles/ising_reranker.py`
**Purpose**: Lightweight pairwise interaction modeling via empirical co-occurrence
**Status**: ✅ Integrated
**Key Classes**: `LightweightIsingReRanker`, `IsingConfig`

### 3. Extremizer Meta-Layer
**File**: `src/modeling/meta/extremizer.py`
**Purpose**: Complete orchestration of calibration + diagnostics + risk classification + reshaping
**Status**: ✅ Integrated
**Key Classes**: `Extremizer`, `ExtremizerConfig`, `PositionIsotonicCalibrator`

## Expected Cumulative Impact

| Phase | Middle-Zone % | Recall@20 % | Status |
|-------|---------------|-------------|--------|
| **Baseline** | 62.8 | 50.92 | ✅ Complete |
| **After Epic 14** | 42-50 | 51-53 | 🎯 Next |
| **After Epic 15** | 38-47 | 51.5-54 | ⏳ Pending |
| **After Epic 16** | 35-43 | 51.5-55 | ⏳ Pending |
| **After Epic 17** | N/A (governance) | N/A | ⏳ Pending |

**Target Achievement**: 35-40% middle-zone (vs 62.8% baseline) = **22-27 pp reduction**

## Project Structure

```
c5-quantum-imputer/
├── docs/
│   ├── prd/
│   │   ├── epic-14-extremizer-meta-layer.md
│   │   ├── epic-15-ising-reranker.md
│   │   ├── epic-16-mixture-of-experts.md
│   │   └── epic-17-governance-observability.md
│   ├── stories/
│   │   └── (stories will be created here: 14.1.*.md, etc.)
│   └── architecture/
│       └── (architecture docs - to be created in Story 14.1)
├── src/
│   └── modeling/
│       ├── calibration/
│       │   ├── __init__.py
│       │   └── position_isotonic.py ✅
│       ├── ensembles/
│       │   ├── __init__.py
│       │   └── ising_reranker.py ✅
│       └── meta/
│           ├── __init__.py
│           └── extremizer.py ✅
└── tests/
    └── modeling/
        ├── calibration/ (to be created)
        ├── ensembles/ (to be created)
        └── meta/ (to be created)
```

## Next Immediate Steps

1. **Create Story 14.1**: Position-Wise Isotonic Calibration Module
   - Use BMAD story template
   - Include architecture context (testing standards, coding standards, tech stack)
   - Create comprehensive Dev Notes section

2. **Begin Epic 14 Implementation**:
   - Story 14.1: Calibration module + tests
   - Story 14.2: Diagnostics engine + tests
   - Story 14.3: Risk classifier + tests
   - Story 14.4: Temperature shaping + tests
   - Story 14.5: Integration & evaluation

3. **Commit Progress**:
   - Commit epic definitions
   - Commit integrated modules
   - Push to GitHub

## Key Architectural Decisions

### Testing Strategy
- **Coverage Target**: ≥90% for all new modules
- **Framework**: pytest + pytest-cov
- **Pattern**: Tests alongside each story (not batched)
- **Validation**: Strict rolling/blocked CV (no temporal leakage)

### Code Standards
- **Style**: PEP8
- **Type Hints**: Required for all functions
- **Docstrings**: Required (Google/NumPy style)
- **Dependencies**: numpy, pandas, scikit-learn, joblib, pyarrow

### Development Workflow
- Each story includes: Implementation + Unit Tests + Integration Test
- Stories marked complete only when ALL acceptance criteria met + tests pass
- Evaluation results logged to `.ai/` directory

## References

- **ChatGPT Audit**: `ChatGPT_response.txt`
- **Project Context**: `START_HERE_AFTER_CHATGPT_AUDIT.md`
- **Comprehensive History**: `COMPREHENSIVE_PROJECT_AUDIT.md`
- **Project Memory**: `.claude/project_memory.md`
- **Archive**: `C:\Users\Minis\CascadeProjects\c5_new-idea` (read-only reference)

---

**Status**: ✅ Infrastructure complete - Ready to begin Story 14.1 development
