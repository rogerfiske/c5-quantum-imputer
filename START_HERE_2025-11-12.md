# START HERE - 2025-11-12 Session

**Date Created**: 2025-11-11 (for 2025-11-12 session)
**Project**: C5 Quantum Imputer - Epic 14-17 Implementation
**Current Epic**: Epic 14 (Extremizer Meta-Layer - Phase A)
**Current Story**: 14.2 (Event-Level Diagnostics Engine) - READY TO START
**Agent**: James (Dev Agent)

---

## 📋 Quick Status

**Where We Are**:
- ✅ Epic 14-17 roadmap complete (20 stories, 4 epics)
- ✅ Architecture documentation complete (4 docs)
- ✅ ChatGPT modules integrated (3 production-ready files)
- ✅ **Story 14.1 COMPLETE** (100% test coverage, ready for review)
- 🎯 **Story 14.2 NEXT** (Event-Level Diagnostics Engine)

**Progress**: 1/20 stories complete (5%) | Epic 14: 1/5 stories (20%)

---

## 🎯 Today's Primary Objective

**Implement Story 14.2: Event-Level Diagnostics Engine**

**Purpose**: Create diagnostics module that computes score shape features to quantify whether event probability distributions are peaked (confident) or diffuse (uncertain).

**Expected Outcome**:
- `src/modeling/meta/diagnostics.py` module with 5 core functions
- Comprehensive unit tests with ≥90% coverage
- Integration test validating diagnostics on synthetic distributions
- All code formatted (black) and linted (flake8)

**Estimated Duration**: 3-4 hours

---

## 📖 Essential Context

### Project Mission
Transform lottery prediction middle-zone distribution from **62.8% (stuck at 2-3 wrong)** to **35-40%** by pushing predictions to extremes (0-1 wrong OR 4-5 wrong).

### Why Story 14.2 Matters
The diagnostics engine is the "brain" that determines whether an event's probability distribution justifies aggressive sharpening (high confidence) or flattening (high uncertainty). This directly enables the middle-zone reduction strategy.

### Current Baseline Performance
- **Recall@20**: 50.92% (1000-event holdout, leak-free)
- **Middle Zone (2-3 wrong)**: 62.8% ← **TARGET TO REDUCE**
- **Excellent (0-1 wrong)**: 20.0%
- **Acceptable (4-5 wrong)**: 17.2%

### Data Characteristics
- **Events**: 11,589 Cash 5 lottery draws
- **Positions**: 39 (5 winners per event)
- **Structure**: 93% random, 7% exploitable patterns
- **Chronic FPs**: QV_12, QV_13, QV_4, QV_14, QV_22

---

## 📂 Story 14.2 Details

**File**: `docs/prd/epic-14-extremizer-meta-layer.md` (Story 14.2 section)

### Acceptance Criteria

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

### Tasks Breakdown

**Task 1**: Implement entropy function
- Shannon entropy: `-Σ(p * log(p))`
- Numerical stability with epsilon clipping

**Task 2**: Implement Gini coefficient
- Gini = (n+1 - 2*Σ(cumulative_p)) / n
- Sort probabilities, compute cumulative sums

**Task 3**: Implement Top-K mass functions
- Top-1 mass, Top-3 mass, Top-5 mass
- Sum of K largest probabilities

**Task 4**: Implement HHI index
- Herfindahl-Hirschman: Σ(p²)
- Measures concentration/diversity

**Task 5**: Implement score gaps
- Compute p[i] - p[i+1] for i=0..4 (5 gaps)
- Measures decisiveness of top predictions

**Task 6**: Write comprehensive unit tests
- Test each function with known inputs/outputs
- Test edge cases (uniform, concentrated, zero probabilities)
- Test numerical stability
- Achieve ≥90% coverage

**Task 7**: Write integration test
- Create synthetic peaked distribution (entropy < 2.0)
- Create synthetic diffuse distribution (entropy > 3.0)
- Validate diagnostics correctly differentiate

### File Locations

**Module**: `src/modeling/meta/diagnostics.py` (create this)
**Tests**: `tests/modeling/meta/test_diagnostics.py` (create this)

---

## 🗺️ Epic 14 Roadmap

```
Epic 14: Extremizer Meta-Layer (Phase A)
Target: -12 to -20 pp middle-zone reduction

Story 14.1: Position Isotonic Calibration       ✅ COMPLETE (100% coverage)
    ↓
Story 14.2: Event-Level Diagnostics            🎯 NEXT (Ready to start)
    ↓
Story 14.3: Middle-Zone Risk Classifier         📝 Defined
    ↓
Story 14.4: Temperature Sharpening/Flattening   📝 Defined
    ↓
Story 14.5: Extremizer Integration & Eval       📝 Defined
```

**After Epic 14**: Epic 15 (Ising Re-Ranker), Epic 16 (MoE), Epic 17 (Governance)

---

## 🔧 Technical Setup

### Dependencies (Already Installed)
- pytest, pytest-cov
- black, flake8
- numpy, pandas, scikit-learn

### Architecture Documents
- **Coding Standards**: `docs/architecture/coding-standards.md`
- **Testing Strategy**: `docs/architecture/testing-strategy.md`
- **Tech Stack**: `docs/architecture/tech-stack.md`
- **Source Tree**: `docs/architecture/source-tree.md`

### Testing Requirements
- **Framework**: pytest
- **Coverage Target**: ≥90%
- **Formatting**: black (line length 100)
- **Linting**: flake8 (0 issues)
- **Type Hints**: Required on all functions
- **Docstrings**: Google-style on all public APIs

---

## 📝 Story 14.2 Implementation Checklist

### Before Starting
- [ ] Read `docs/prd/epic-14-extremizer-meta-layer.md` (Story 14.2 section)
- [ ] Review Story 14.1 completion: `docs/stories/14.1.position-isotonic-calibration.md`
- [ ] Review `docs/architecture/testing-strategy.md` for test patterns
- [ ] Ensure `src/modeling/meta/__init__.py` exists

### During Implementation
- [ ] Create `src/modeling/meta/diagnostics.py`
- [ ] Implement 5 diagnostic functions (entropy, Gini, top-K, HHI, gaps)
- [ ] Add type hints and docstrings to all functions
- [ ] Create `tests/modeling/meta/test_diagnostics.py`
- [ ] Write unit tests for each function (≥6 tests per function)
- [ ] Write integration test (peaked vs diffuse)
- [ ] Run pytest with coverage: `pytest tests/modeling/meta/test_diagnostics.py -v --cov=src.modeling.meta.diagnostics --cov-report=term`
- [ ] Achieve ≥90% coverage
- [ ] Format with black: `black src/modeling/meta/diagnostics.py tests/modeling/meta/test_diagnostics.py`
- [ ] Lint with flake8: `flake8 src/modeling/meta/diagnostics.py tests/modeling/meta/test_diagnostics.py --max-line-length=100`

### Story Completion
- [ ] Update story file tasks to [x]
- [ ] Update Dev Agent Record section
- [ ] Update story status to "Ready for Review"
- [ ] Commit to git with descriptive message
- [ ] Proceed to Story 14.3 or take break

---

## 📚 Key Reference Files

### Story Files
- **Current Story**: `docs/stories/14.2.*.md` (will be created using BMAD template)
- **Previous Story**: `docs/stories/14.1.position-isotonic-calibration.md` (reference for patterns)

### Epic Definitions
- **Epic 14**: `docs/prd/epic-14-extremizer-meta-layer.md`
- **Epic 15**: `docs/prd/epic-15-ising-reranker.md`
- **Epic 16**: `docs/prd/epic-16-mixture-of-experts.md`
- **Epic 17**: `docs/prd/epic-17-governance-observability.md`

### Implementation Roadmap
- **EPIC_14-17_ROADMAP.md**: Complete 4-epic overview with expected impacts

### Session History
- **SESSION_SUMMARY_2025-11-11.md**: Yesterday's accomplishments and learnings

### ChatGPT Audit
- **ChatGPT_response.txt**: Original audit response with recommendations
- **COMPREHENSIVE_PROJECT_AUDIT.md**: 60-page experimental history

---

## 🚀 How to Start Story 14.2

### Option A: Use Dev Agent Command (Recommended)

```
*develop-story
```

This will:
1. Load Story 14.2 (after creation)
2. Execute tasks sequentially
3. Write tests alongside implementation
4. Track progress in todo list
5. Update story file upon completion

### Option B: Manual Implementation

1. Create Story 14.2 file using BMAD template
2. Implement `src/modeling/meta/diagnostics.py`
3. Write comprehensive tests
4. Run coverage and quality checks
5. Update story file and commit

---

## ⚠️ Important Reminders

### From Story 14.1 Learnings

1. **Tests Drive Coverage**: CLI tests were critical for 100% coverage
2. **Synthetic Data Quality**: Use `np.random.seed()` for reproducibility
3. **Numerical Stability**: Always clip probabilities, handle division by zero
4. **Vectorization**: Use NumPy operations, avoid Python loops
5. **Documentation**: Type hints and docstrings catch issues early

### Common Pitfalls to Avoid

- ❌ Don't loop over events in Python (use vectorized NumPy)
- ❌ Don't forget epsilon clipping for log operations
- ❌ Don't skip integration tests (peaked vs diffuse validation)
- ❌ Don't commit without running black + flake8
- ❌ Don't proceed to Story 14.3 until 14.2 is "Ready for Review"

---

## 📊 Expected Diagnostics Output Example

**Input**: Calibrated probabilities (n_events, 39)

```python
# Example: 3 events
p = np.array([
    [0.10, 0.08, 0.06, 0.05, 0.04, ...],  # Event 1 (moderately peaked)
    [0.25, 0.20, 0.15, 0.10, 0.05, ...],  # Event 2 (highly peaked)
    [0.026, 0.026, 0.026, 0.025, ...],    # Event 3 (diffuse/uniform)
])
```

**Output**: Diagnostics (n_events, 11)

```python
# Columns: [entropy, gini, hhi, top1, top3, top5, gap1, gap2, gap3, gap4, gap5]
diagnostics = np.array([
    [2.8, 0.45, 0.08, 0.10, 0.24, 0.35, 0.02, 0.02, 0.01, 0.01, 0.01],  # Event 1
    [2.0, 0.60, 0.15, 0.25, 0.60, 0.75, 0.05, 0.05, 0.05, 0.05, 0.02],  # Event 2
    [3.6, 0.20, 0.03, 0.026, 0.078, 0.130, 0.0, 0.0, 0.001, 0.0, 0.0],  # Event 3
])
```

**Interpretation**:
- Event 2: Low entropy (2.0), high Gini (0.60), large gaps → **Sharpen** (confident)
- Event 3: High entropy (3.6), low Gini (0.20), small gaps → **Flatten** (uncertain)

---

## 🎯 Success Criteria for Today

### Minimum Viable Completion (Story 14.2)
- [ ] `diagnostics.py` module created with 5 functions
- [ ] All functions have type hints and docstrings
- [ ] ≥30 unit tests created (≥6 per function)
- [ ] Test coverage ≥90%
- [ ] Integration test validates peaked vs diffuse
- [ ] Code passes black + flake8
- [ ] Story status: "Ready for Review"
- [ ] Git commit pushed

### Stretch Goals
- [ ] Test coverage 100%
- [ ] Begin Story 14.3 (Middle-Zone Risk Classifier)
- [ ] Document any edge cases or numerical issues discovered

---

## 💬 Questions to Resolve Today

### Technical Decisions
- Q: Should epsilon be configurable or hardcoded (1e-12)?
  - Recommendation: Configurable with sensible default

- Q: Should diagnostics module be standalone or part of extremizer.py?
  - Recommendation: Standalone (better testability, follows Story 14.2 spec)

- Q: How to handle all-zero probability vectors?
  - Recommendation: Return NaN or default values, document in docstring

### Process Questions
- Q: Create Story 14.2 file manually or have agent create it?
  - Recommendation: Agent creates using BMAD template (consistent format)

---

## 📞 Support Resources

**If Stuck**:
1. Review `SESSION_SUMMARY_2025-11-11.md` for patterns from Story 14.1
2. Check `docs/architecture/testing-strategy.md` for test patterns
3. Reference Story 14.1 test file: `tests/modeling/calibration/test_position_isotonic.py`
4. Review ChatGPT audit: `ChatGPT_response.txt` (Detailed Audit section)

**Git Status Check**:
```bash
git status
git log -1 --oneline
```

**Test Execution**:
```bash
# Run Story 14.2 tests
pytest tests/modeling/meta/test_diagnostics.py -v --cov=src.modeling.meta.diagnostics --cov-report=term

# Run all tests
pytest tests/ -v --cov=src --cov-report=term
```

---

## 📅 Upcoming Milestones

**This Week**:
- Story 14.2 (Event-Level Diagnostics) ← **TODAY**
- Story 14.3 (Middle-Zone Risk Classifier)

**Next Week**:
- Story 14.4 (Temperature Sharpening/Flattening)
- Story 14.5 (Extremizer Integration & Evaluation)
- Begin Epic 15 (Ising Re-Ranker)

**Month Goal**: Complete Epic 14-15 (10 stories), validate middle-zone reduction

---

## 🎬 Session Start Command

When ready to begin:

```
*develop-story
```

Or for manual start:
1. Load Story 14.2 definition from Epic 14
2. Create story file using BMAD template
3. Begin Task 1 (Implement entropy function)

---

**Good luck with Story 14.2! 🚀**

**Agent**: James (Dev Agent) - Ready to continue systematic Epic 14 implementation
**Target**: Middle-zone 62.8% → 35-40% = -22 to -27 pp reduction
**Approach**: Test-driven, vectorized, numerically stable implementations

---

**File Created**: 2025-11-11
**For Session**: 2025-11-12
**Current Status**: Story 14.1 complete, Story 14.2 ready to start
