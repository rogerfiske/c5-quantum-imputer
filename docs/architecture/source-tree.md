# Source Tree - C5 Quantum Imputer

**Version**: 1.0
**Last Updated**: 2025-11-11

## Project Root Structure

```
c5-quantum-imputer/
├── .bmad-core/                         # BMAD framework configuration
│   ├── checklists/                     # Story DoD, deployment checklists
│   ├── tasks/                          # Reusable task definitions
│   ├── templates/                      # Story, epic, architecture templates
│   └── core-config.yaml                # Project configuration
│
├── .claude/                            # Claude Code configuration
│   ├── commands/                       # Custom slash commands
│   ├── project_memory.md               # Project context for Claude
│   └── settings.local.json             # Local settings
│
├── .ai/                                # AI-generated outputs
│   ├── debug-log.md                    # Development debug logs
│   └── epic-*-evaluation-results.md    # Per-epic evaluation reports
│
├── .github/                            # GitHub configuration
│   └── workflows/                      # CI/CD workflows (Epic 17)
│
├── docs/                               # Documentation
│   ├── prd/                            # Product requirements (Epics)
│   │   ├── epic-14-extremizer-meta-layer.md
│   │   ├── epic-15-ising-reranker.md
│   │   ├── epic-16-mixture-of-experts.md
│   │   └── epic-17-governance-observability.md
│   ├── stories/                        # Story files (BMAD format)
│   │   └── {epic}.{story}.*.md         # Format: 14.1.position-isotonic.md
│   └── architecture/                   # Architecture documentation
│       ├── tech-stack.md               # Technology decisions
│       ├── coding-standards.md         # Code style guide
│       ├── testing-strategy.md         # Testing approach
│       └── source-tree.md              # This file
│
├── data/                               # Dataset storage
│   ├── raw/                            # Original data
│   │   └── c5_Matrix.csv               # 11,589 Cash 5 lottery events
│   └── processed/                      # Processed datasets (parquet)
│
├── src/                                # Source code
│   ├── __init__.py
│   ├── imputation/                     # Quantum imputation methods
│   │   ├── __init__.py
│   │   ├── amplitude_embedding.py      # Primary method (242 features)
│   │   ├── density_matrix.py           # Alternative method
│   │   ├── angle_encoding.py           # Alternative method
│   │   └── graph_cycle.py              # Alternative method
│   │
│   ├── modeling/                       # ML modeling layer
│   │   ├── __init__.py
│   │   ├── calibration/                # **NEW: Epic 14**
│   │   │   ├── __init__.py
│   │   │   └── position_isotonic.py    # Per-position isotonic calibration
│   │   │
│   │   ├── ensembles/                  # Ensemble methods
│   │   │   ├── __init__.py
│   │   │   └── ising_reranker.py       # **NEW: Epic 15** - Pairwise interactions
│   │   │
│   │   ├── meta/                       # **NEW: Epic 14** - Meta-learning
│   │   │   ├── __init__.py
│   │   │   ├── extremizer.py           # Main extremizer orchestrator
│   │   │   ├── diagnostics.py          # Event-level diagnostics (Story 14.2)
│   │   │   ├── risk_classifier.py      # Middle-zone risk classifier (Story 14.3)
│   │   │   └── temperature_shaping.py  # Gamma-based reshaping (Story 14.4)
│   │   │
│   │   └── rankers/                    # Ranking models
│   │       ├── __init__.py
│   │       └── lgbm_ranker.py          # LightGBM baseline ranker
│   │
│   ├── evaluation/                     # Evaluation utilities
│   │   ├── __init__.py
│   │   └── metrics.py                  # Custom metrics (Recall@20, distribution)
│   │
│   ├── monitoring/                     # **NEW: Epic 17** - Observability
│   │   ├── __init__.py
│   │   ├── change_point_detection.py   # BOCPD for regime shifts
│   │   ├── calibration_drift.py        # Drift monitoring
│   │   ├── subgroup_rules.py           # Interpretable rule discovery
│   │   ├── logger.py                   # Event logging
│   │   └── dashboard.py                # Monitoring dashboard
│   │
│   └── utils/                          # Shared utilities
│       ├── __init__.py
│       └── data_loader.py              # Data loading helpers
│
├── tests/                              # Test suite (mirrors src/ structure)
│   ├── __init__.py
│   ├── modeling/
│   │   ├── calibration/
│   │   │   └── test_position_isotonic.py       # Story 14.1 tests
│   │   ├── meta/
│   │   │   ├── test_diagnostics.py              # Story 14.2 tests
│   │   │   ├── test_risk_classifier.py          # Story 14.3 tests
│   │   │   ├── test_temperature_shaping.py      # Story 14.4 tests
│   │   │   └── test_extremizer.py               # Story 14.5 tests
│   │   └── ensembles/
│   │       └── test_ising_reranker.py           # Story 15.4 tests
│   ├── integration/
│   │   ├── test_epic14_pipeline.py              # Epic 14 integration
│   │   └── test_epic15_pipeline.py              # Epic 15 integration
│   ├── e2e/
│   │   └── test_full_pipeline_e2e.py            # End-to-end holdout test
│   └── fixtures/                                # Test data fixtures
│
├── scripts/                            # Standalone scripts
│   ├── evaluate_extremizer.py          # Story 14.5 evaluation
│   ├── evaluate_ising_reranker.py      # Story 15.5 evaluation
│   ├── evaluate_moe.py                 # Story 16.5 evaluation
│   ├── generate_dashboard.py           # Story 17.4 dashboard
│   └── recalibrate_model.py            # Story 17.5 recalibration
│
├── production/                         # Production models and artifacts
│   ├── models/
│   │   ├── amplitude/
│   │   │   └── lgbm_ranker_baseline_fixed_v1.pkl  # Baseline LGBM
│   │   ├── calibrators/                           # **NEW: Epic 14**
│   │   └── extremizers/                           # **NEW: Epic 14**
│   ├── comprehensive_1000event_holdout_diagnostic.py  # Holdout test
│   └── predict_next_event.py                          # Production prediction
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore patterns
├── LICENSE                             # Project license
├── README.md                           # Project overview
├── COMPREHENSIVE_PROJECT_AUDIT.md      # 60+ page experimental history
├── DATASET_ANALYSIS.md                 # Statistical analysis
├── START_HERE_AFTER_CHATGPT_AUDIT.md   # Transition document
├── ChatGPT_response.txt                # ChatGPT audit recommendations
└── EPIC_14-17_ROADMAP.md               # Epic implementation roadmap
```

## Module Responsibility Map

### src/imputation/
**Purpose**: Transform sparse binary lottery data into rich quantum feature representations
**Key Files**: `amplitude_embedding.py` (primary, 242 features)
**Dependencies**: qiskit, numpy

### src/modeling/calibration/ (NEW - Epic 14)
**Purpose**: Per-position calibration to reduce chronic false positives
**Key Files**: `position_isotonic.py`
**Dependencies**: scikit-learn (IsotonicRegression)

### src/modeling/meta/ (NEW - Epic 14)
**Purpose**: Meta-learning layer for event-level decision making
**Key Files**: `extremizer.py` (orchestrator), `diagnostics.py`, `risk_classifier.py`, `temperature_shaping.py`
**Dependencies**: scikit-learn (LogisticRegression), numpy

### src/modeling/ensembles/
**Purpose**: Ensemble methods and pairwise interaction modeling
**Key Files**: `ising_reranker.py` (NEW - Epic 15)
**Dependencies**: numpy, pandas

### src/modeling/rankers/
**Purpose**: Core ranking models
**Key Files**: `lgbm_ranker.py` (baseline)
**Dependencies**: lightgbm

### src/monitoring/ (NEW - Epic 17)
**Purpose**: Observability, drift detection, and governance
**Key Files**: `change_point_detection.py`, `calibration_drift.py`, `subgroup_rules.py`
**Dependencies**: scikit-learn, matplotlib, plotly

## File Naming Conventions

### Source Files
- **Modules**: `snake_case.py` (e.g., `position_isotonic.py`)
- **Classes**: PascalCase (e.g., `PositionIsotonicCalibrator`)
- **Test files**: `test_{module_name}.py` (e.g., `test_position_isotonic.py`)

### Data Files
- **Raw data**: CSV format (`c5_Matrix.csv`)
- **Processed data**: Parquet format (`calibrated_scores.parquet`)
- **Models**: `.pkl` or `.joblib` (e.g., `lgbm_ranker_baseline_fixed_v1.pkl`)

### Documentation Files
- **Epics**: `epic-{N}-{slug}.md` (e.g., `epic-14-extremizer-meta-layer.md`)
- **Stories**: `{epic}.{story}.{slug}.md` (e.g., `14.1.position-isotonic.md`)
- **Architecture**: `{topic}.md` (e.g., `tech-stack.md`)

## Import Paths

### Absolute Imports (Preferred)
```python
from src.modeling.calibration.position_isotonic import PositionIsotonicCalibrator
from src.modeling.meta.extremizer import Extremizer
from src.evaluation.metrics import recall_at_20
```

### Relative Imports (Within Package)
```python
# Within src/modeling/meta/ package
from .diagnostics import entropy, gini_coefficient
from .risk_classifier import MiddleZoneRiskClassifier
```

## Data Flow Architecture

```
Raw CSV Data (data/raw/c5_Matrix.csv)
    ↓
[Imputation] (src/imputation/amplitude_embedding.py)
    ↓
Quantum Features (242 dimensions)
    ↓
[LGBM Ranker] (src/modeling/rankers/lgbm_ranker.py)
    ↓
Raw Scores (39 positions)
    ↓
[Calibration] (src/modeling/calibration/position_isotonic.py)
    ↓
Calibrated Probabilities
    ↓
[Extremizer] (src/modeling/meta/extremizer.py)
    ↓
Sharpened/Flattened Probabilities
    ↓
[Ising Re-Ranker] (src/modeling/ensembles/ising_reranker.py) [Optional]
    ↓
Final Probabilities
    ↓
Top-20 Selection
```

## Directory Ownership

| Directory | Primary Owner | Epic |
|-----------|---------------|------|
| `src/modeling/calibration/` | Dev Agent | 14 |
| `src/modeling/meta/` | Dev Agent | 14 |
| `src/modeling/ensembles/` | Dev Agent | 15 |
| `src/monitoring/` | Dev Agent | 17 |
| `tests/modeling/` | Dev Agent | 14-16 |
| `tests/integration/` | Dev Agent | 14-17 |
| `scripts/` | Dev Agent | 14-17 |
| `docs/stories/` | Scrum Master | All |
| `docs/architecture/` | Architect | All |

## Archive Reference

**Location**: `C:\Users\Minis\CascadeProjects\c5_new-idea` (READ-ONLY)
**Purpose**: Historical experimental scripts and failed approaches (Epic 1-13)
**When to reference**: Understanding "why didn't X work?" or reusing diagnostic utilities

## Notes

- **Epic 14-17** introduce new directories: `calibration/`, `meta/`, `monitoring/`
- **Tests mirror src/ structure** for easy navigation
- **All new code requires ≥90% test coverage**
- **Integration tests** validate cross-module pipelines
- **Production artifacts** stored in `production/models/` subdirectories by method
