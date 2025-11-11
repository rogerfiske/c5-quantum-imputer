"""
Comprehensive Holdout Test with Deep Diagnostics

This script runs a complete holdout test on the FIXED baseline model
and collects extensive metrics to identify:
1. Where predictions are stuck in the middle (2-3 wrong) vs extremes (0-1, 4-5 wrong)
2. Which positions/QVs need stronger signals
3. Script calculations/algorithms that can be modified to force extreme outcomes
4. Actionable improvement paths

Based on Diagnostic Tools 1-4 from previous work.

Usage:
    python production/comprehensive_1000event_holdout_diagnostic.py [num_events]

Example:
    python production/comprehensive_1000event_holdout_diagnostic.py 20   # Test on 20 events
    python production/comprehensive_1000event_holdout_diagnostic.py 1000 # Full test
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from collections import Counter, defaultdict
from datetime import datetime

# Fix Windows console encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.imputation.amplitude_embedding import AmplitudeEmbedding

# Get number of events from command line (default: 1000)
num_test_events = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

print("=" * 80)
print(f"COMPREHENSIVE {num_test_events}-EVENT HOLDOUT DIAGNOSTIC")
print("Fixed Architecture (NO LEAKAGE) - Deep Analysis")
print("=" * 80)

# ============================================================================
# SECTION 1: LOAD DATA AND MODEL
# ============================================================================

print(f"\n[1/8] Loading data and model...")

# Load data
raw_data_path = project_root / "data" / "raw" / "c5_Matrix.csv"
df_raw = pd.read_csv(raw_data_path)
train_df = df_raw[df_raw['event-ID'] <= 10589].copy()
holdout_df = df_raw[(df_raw['event-ID'] >= 10590) & (df_raw['event-ID'] <= 11589)].copy()

# Use specified number of events
holdout_df = holdout_df.head(num_test_events)

num_events_actual = len(holdout_df)
print(f"  Train: {len(train_df)} events (event-ID 1 to 10589)")
print(f"  Holdout: {num_events_actual} events (event-ID {holdout_df['event-ID'].iloc[0]} to {holdout_df['event-ID'].iloc[-1]})")

# Load fixed baseline model
model_path = project_root / "production" / "models" / "amplitude" / "lgbm_ranker_baseline_fixed_v1.pkl"
if not model_path.exists():
    print(f"\n[ERROR] Model not found: {model_path}")
    print(f"[ERROR] Please run story_12_4_retrain_fixed_baseline.py first")
    sys.exit(1)

ranker = joblib.load(model_path)
print(f"  [OK] Model loaded: lgbm_ranker_baseline_fixed_v1.pkl")

# Initialize imputer
imputer = AmplitudeEmbedding(
    include_proximity_features=False,
    include_low_boundary_features=False
)
imputer.fit(train_df)
print(f"  [OK] Imputer fitted")

# ============================================================================
# SECTION 2: RUN HOLDOUT TEST WITH DETAILED METRICS
# ============================================================================

print(f"\n[2/8] Running {num_events_actual}-event sequential holdout test...")
print(f"  Collecting detailed metrics for each prediction...")

# Data structures for comprehensive metrics
num_wrong_per_event = []
qv_cols = [f'QV_{i}' for i in range(1, 40)]

# Per-event detailed tracking
event_details = []

# Per-QV performance tracking
qv_performance = defaultdict(lambda: {
    'total_appearances': 0,
    'predicted_correctly': 0,
    'predicted_incorrectly': 0,
    'missed_entirely': 0,
    'false_positives': 0,
    'imputed_prob_when_correct': [],
    'imputed_prob_when_wrong': [],
    'rank_when_correct': [],
    'rank_when_wrong': [],
})

# Position-specific confidence analysis
position_confidence = defaultdict(list)

# Wrong distribution by QV difficulty (we'll categorize later)
wrong_by_difficulty = defaultdict(lambda: defaultdict(int))

print(f"  Progress: ", end='', flush=True)
progress_interval = 100

for idx, (i, event_row) in enumerate(holdout_df.iterrows()):
    if (idx + 1) % progress_interval == 0:
        print(f"{idx + 1}...", end='', flush=True)

    # Historical context: train + holdout up to (but NOT including) current event
    holdout_so_far = holdout_df.iloc[:idx] if idx > 0 else pd.DataFrame()
    historical_context = pd.concat([train_df, holdout_so_far]) if len(holdout_so_far) > 0 else train_df

    # Get imputed probabilities (for diagnostics)
    imputed_probs = imputer._impute_next_event_probabilities(historical_context)

    # Predict using CORRECT interface (NO LEAKAGE)
    top20 = imputer.predict_with_imputation(
        historical_context,
        ranker_model=ranker,
        num_predictions=20
    )

    # Get actual winning positions
    actual_positions = [i+1 for i, val in enumerate(event_row[qv_cols].values) if val == 1]

    # Calculate accuracy
    correct_predictions = [pos for pos in actual_positions if pos in top20]
    num_correct = len(correct_predictions)
    num_wrong = 5 - num_correct
    num_wrong_per_event.append(num_wrong)

    # Detailed event record
    event_detail = {
        'event_id': int(event_row['event-ID']),
        'actual_positions': actual_positions,
        'top20_predictions': top20,
        'num_correct': num_correct,
        'num_wrong': num_wrong,
        'correct_positions': correct_predictions,
        'missed_positions': [pos for pos in actual_positions if pos not in top20],
        'false_positives': [pos for pos in top20 if pos not in actual_positions],
    }
    event_details.append(event_detail)

    # Per-QV performance tracking
    for qv_pos in range(1, 40):
        qv_idx = qv_pos - 1
        is_actual_winner = event_row[f'QV_{qv_pos}'] == 1
        is_predicted = qv_pos in top20

        if is_actual_winner:
            qv_performance[qv_pos]['total_appearances'] += 1

            if is_predicted:
                qv_performance[qv_pos]['predicted_correctly'] += 1
                qv_performance[qv_pos]['imputed_prob_when_correct'].append(imputed_probs[qv_idx])
                qv_performance[qv_pos]['rank_when_correct'].append(top20.index(qv_pos) + 1)
            else:
                qv_performance[qv_pos]['missed_entirely'] += 1
                qv_performance[qv_pos]['imputed_prob_when_wrong'].append(imputed_probs[qv_idx])
        else:
            if is_predicted:
                qv_performance[qv_pos]['false_positives'] += 1

        # Track confidence (imputed probability) for all positions
        position_confidence[qv_pos].append({
            'event_id': int(event_row['event-ID']),
            'imputed_prob': imputed_probs[qv_idx],
            'was_actual': is_actual_winner,
            'was_predicted': is_predicted,
            'rank': top20.index(qv_pos) + 1 if is_predicted else None,
        })

print(f" DONE")

# ============================================================================
# SECTION 3: CALCULATE PRIMARY METRICS
# ============================================================================

print(f"\n[3/8] Calculating primary metrics...")

total_events = len(num_wrong_per_event)
wrong_counts = Counter(num_wrong_per_event)
total_correct = sum(5 - w for w in num_wrong_per_event)
total_positions = total_events * 5
overall_recall = total_correct / total_positions

print(f"  Events analyzed: {total_events}")
print(f"  Overall recall@20: {overall_recall:.2%} ({total_correct}/{total_positions})")

# Calculate extremes vs middle distribution
excellent_good = wrong_counts.get(0, 0) + wrong_counts.get(1, 0)
poor = wrong_counts.get(2, 0) + wrong_counts.get(3, 0)
acceptable = wrong_counts.get(4, 0) + wrong_counts.get(5, 0)

excellent_good_pct = (excellent_good / total_events) * 100
poor_pct = (poor / total_events) * 100
acceptable_pct = (acceptable / total_events) * 100

print(f"\n  Distribution Analysis:")
print(f"    Excellent/Good (0-1 wrong): {excellent_good_pct:.1f}% ← TARGET: INCREASE")
print(f"    Poor (2-3 wrong):           {poor_pct:.1f}% ← TARGET: DECREASE")
print(f"    Acceptable (4-5 wrong):     {acceptable_pct:.1f}% ← ACCEPTABLE")

# ============================================================================
# SECTION 4: QV DIFFICULTY ANALYSIS
# ============================================================================

print(f"\n[4/8] Analyzing QV-specific performance...")

# Calculate recall@20 for each QV
qv_metrics = {}
for qv_pos in range(1, 40):
    perf = qv_performance[qv_pos]
    if perf['total_appearances'] > 0:
        recall = perf['predicted_correctly'] / perf['total_appearances']
        avg_prob_correct = np.mean(perf['imputed_prob_when_correct']) if perf['imputed_prob_when_correct'] else 0
        avg_prob_wrong = np.mean(perf['imputed_prob_when_wrong']) if perf['imputed_prob_when_wrong'] else 0
        avg_rank_correct = np.mean(perf['rank_when_correct']) if perf['rank_when_correct'] else 0

        qv_metrics[qv_pos] = {
            'appearances': perf['total_appearances'],
            'recall': recall,
            'avg_imputed_prob_when_correct': avg_prob_correct,
            'avg_imputed_prob_when_wrong': avg_prob_wrong,
            'avg_rank_when_correct': avg_rank_correct,
            'false_positive_rate': perf['false_positives'] / total_events,
        }
    else:
        qv_metrics[qv_pos] = {
            'appearances': 0,
            'recall': 0,
            'avg_imputed_prob_when_correct': 0,
            'avg_imputed_prob_when_wrong': 0,
            'avg_rank_when_correct': 0,
            'false_positive_rate': 0,
        }

# Categorize QVs by difficulty (based on recall)
difficulty_categories = {
    'EXCELLENT': [],  # recall >= 0.8
    'GOOD': [],       # 0.6 <= recall < 0.8
    'MEDIUM': [],     # 0.4 <= recall < 0.6
    'POOR': [],       # 0.2 <= recall < 0.4
    'CRITICAL': [],   # recall < 0.2
}

for qv_pos, metrics in qv_metrics.items():
    if metrics['appearances'] < 10:  # Skip rare QVs
        continue

    recall = metrics['recall']
    if recall >= 0.8:
        difficulty_categories['EXCELLENT'].append(qv_pos)
    elif recall >= 0.6:
        difficulty_categories['GOOD'].append(qv_pos)
    elif recall >= 0.4:
        difficulty_categories['MEDIUM'].append(qv_pos)
    elif recall >= 0.2:
        difficulty_categories['POOR'].append(qv_pos)
    else:
        difficulty_categories['CRITICAL'].append(qv_pos)

print(f"  QV Difficulty Categories:")
for category, qvs in difficulty_categories.items():
    if qvs:
        print(f"    {category:10s}: {len(qvs):2d} QVs - {qvs[:10]}{' ...' if len(qvs) > 10 else ''}")

# ============================================================================
# SECTION 5: CONFIDENCE ANALYSIS (Key to Forcing Extremes!)
# ============================================================================

print(f"\n[5/8] Analyzing confidence patterns...")

# Analyze where predictions are "wishy-washy" (medium confidence)
confidence_ranges = {
    'very_high': (0.8, 1.0),    # Should predict confidently
    'high': (0.6, 0.8),
    'medium': (0.4, 0.6),       # PROBLEM ZONE - wishy-washy
    'low': (0.2, 0.4),
    'very_low': (0.0, 0.2),     # Should reject confidently
}

confidence_analysis = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total': 0})

for qv_pos in range(1, 40):
    for conf_record in position_confidence[qv_pos]:
        prob = conf_record['imputed_prob']
        was_actual = conf_record['was_actual']
        was_predicted = conf_record['was_predicted']

        # Determine confidence range
        for range_name, (low, high) in confidence_ranges.items():
            if low <= prob < high:
                confidence_analysis[range_name]['total'] += 1

                # Was this a good decision?
                if was_actual and was_predicted:
                    confidence_analysis[range_name]['correct'] += 1
                elif not was_actual and not was_predicted:
                    confidence_analysis[range_name]['correct'] += 1
                else:
                    confidence_analysis[range_name]['wrong'] += 1
                break

print(f"  Confidence Range Performance:")
for range_name in ['very_high', 'high', 'medium', 'low', 'very_low']:
    stats = confidence_analysis[range_name]
    if stats['total'] > 0:
        accuracy = stats['correct'] / stats['total']
        print(f"    {range_name:10s} ({confidence_ranges[range_name][0]:.1f}-{confidence_ranges[range_name][1]:.1f}): "
              f"{stats['total']:6d} decisions, {accuracy:5.1%} accurate")

# ============================================================================
# SECTION 6: IDENTIFY IMPROVEMENT TARGETS
# ============================================================================

print(f"\n[6/8] Identifying improvement targets...")

# Find positions stuck in "medium confidence" zone
medium_confidence_positions = []
for qv_pos in range(1, 40):
    medium_count = 0
    total_count = 0
    for conf_record in position_confidence[qv_pos]:
        prob = conf_record['imputed_prob']
        if 0.4 <= prob < 0.6:
            medium_count += 1
        total_count += 1

    if total_count > 0:
        medium_pct = medium_count / total_count
        if medium_pct > 0.3:  # More than 30% in medium range
            medium_confidence_positions.append((qv_pos, medium_pct, qv_metrics[qv_pos]['recall']))

# Sort by medium confidence percentage
medium_confidence_positions.sort(key=lambda x: x[1], reverse=True)

print(f"  Positions stuck in MEDIUM confidence (0.4-0.6):")
print(f"  These need stronger signals to push to extremes!")
for qv_pos, medium_pct, recall in medium_confidence_positions[:15]:
    print(f"    QV {qv_pos:2d}: {medium_pct:5.1%} in medium zone, recall={recall:5.1%}")

# Find positions with high false positive rate
high_fp_positions = [(qv, m['false_positive_rate'], m['recall'])
                     for qv, m in qv_metrics.items()
                     if m['false_positive_rate'] > 0.3]
high_fp_positions.sort(key=lambda x: x[1], reverse=True)

if high_fp_positions:
    print(f"\n  Positions with high FALSE POSITIVE rate:")
    print(f"  These are predicted too often when wrong!")
    for qv_pos, fp_rate, recall in high_fp_positions[:10]:
        print(f"    QV {qv_pos:2d}: FP rate={fp_rate:5.1%}, recall={recall:5.1%}")

# ============================================================================
# SECTION 7: ACTIONABLE RECOMMENDATIONS
# ============================================================================

print(f"\n[7/8] Generating actionable recommendations...")

recommendations = {
    'confidence_boosting': [],
    'feature_enhancements': [],
    'algorithm_modifications': [],
    'imputation_improvements': [],
}

# Recommendation 1: Confidence thresholding
medium_zone_pct = (confidence_analysis['medium']['total'] / sum(c['total'] for c in confidence_analysis.values())) * 100
if medium_zone_pct > 20:
    recommendations['algorithm_modifications'].append({
        'priority': 'HIGH',
        'target': 'Confidence Thresholding',
        'problem': f'{medium_zone_pct:.1f}% of predictions in medium confidence zone (0.4-0.6)',
        'solution': 'Apply confidence boosting: prob < 0.3 → force to 0, prob > 0.7 → boost to 1.0',
        'expected_impact': 'Push 20-30% more predictions to extremes (0-1 or 4-5 wrong)',
        'implementation': 'Modify _impute_next_event_probabilities() to apply sigmoid boosting',
    })

# Recommendation 2: Position-specific weights
if len(medium_confidence_positions) > 5:
    recommendations['feature_enhancements'].append({
        'priority': 'HIGH',
        'target': 'Position-Specific Features',
        'problem': f'{len(medium_confidence_positions)} positions stuck in medium confidence',
        'solution': 'Add LOW_BOUNDARY features (from Epic 11.1) but with fixed architecture',
        'expected_impact': 'Improve recall for QVs 2, 3, 10, 11, 14, 15 by 10-20%',
        'implementation': 'Enable include_low_boundary_features=True in imputer',
    })

# Recommendation 3: Temporal weighting
recommendations['imputation_improvements'].append({
    'priority': 'MEDIUM',
    'target': 'Temporal Weighting in Imputation',
    'problem': 'Simple frequency treats all 50 events equally',
    'solution': 'Weight recent events more heavily (exponential decay)',
    'expected_impact': 'Better capture trends, improve recall by 5-10%',
    'implementation': 'Modify _impute_next_event_probabilities() to apply decay weights',
})

# Recommendation 4: Ensemble voting
recommendations['algorithm_modifications'].append({
    'priority': 'MEDIUM',
    'target': 'Ensemble Confidence Voting',
    'problem': 'Single model produces middle-ground predictions',
    'solution': 'Use multiple imputation methods, vote only on high-agreement positions',
    'expected_impact': 'Force more predictions to extremes (high confidence = agree, low = disagree)',
    'implementation': 'Create ensemble of: frequency, pattern-based, temporal-weighted',
})

print(f"  Generated {sum(len(v) for v in recommendations.values())} recommendations")
print(f"  Priorities: HIGH={sum(1 for recs in recommendations.values() for r in recs if r['priority']=='HIGH')}, "
      f"MEDIUM={sum(1 for recs in recommendations.values() for r in recs if r['priority']=='MEDIUM')}")

# ============================================================================
# SECTION 8: SAVE COMPREHENSIVE RESULTS
# ============================================================================

print(f"\n[8/8] Saving comprehensive results...")

# Create output directory
output_dir = project_root / "production" / "reports" / f"comprehensive_{num_events_actual}event_diagnostic"
output_dir.mkdir(parents=True, exist_ok=True)

# Save summary report (JSON)
summary = {
    'test_metadata': {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'lgbm_ranker_baseline_fixed_v1.pkl',
        'architecture': 'Fixed (NO LEAKAGE) - Two-stage prediction',
        'events_tested': total_events,
        'event_range': [int(holdout_df['event-ID'].iloc[0]), int(holdout_df['event-ID'].iloc[-1])],
    },
    'primary_metrics': {
        'overall_recall_at_20': overall_recall,
        'total_correct': total_correct,
        'total_positions': total_positions,
    },
    'wrong_distribution': {
        '0_wrong': {'count': wrong_counts.get(0, 0), 'percentage': (wrong_counts.get(0, 0) / total_events) * 100},
        '1_wrong': {'count': wrong_counts.get(1, 0), 'percentage': (wrong_counts.get(1, 0) / total_events) * 100},
        '2_wrong': {'count': wrong_counts.get(2, 0), 'percentage': (wrong_counts.get(2, 0) / total_events) * 100},
        '3_wrong': {'count': wrong_counts.get(3, 0), 'percentage': (wrong_counts.get(3, 0) / total_events) * 100},
        '4_wrong': {'count': wrong_counts.get(4, 0), 'percentage': (wrong_counts.get(4, 0) / total_events) * 100},
        '5_wrong': {'count': wrong_counts.get(5, 0), 'percentage': (wrong_counts.get(5, 0) / total_events) * 100},
    },
    'extreme_vs_middle': {
        'excellent_good_0_1_wrong': {'count': excellent_good, 'percentage': excellent_good_pct},
        'poor_2_3_wrong': {'count': poor, 'percentage': poor_pct},
        'acceptable_4_5_wrong': {'count': acceptable, 'percentage': acceptable_pct},
    },
    'qv_difficulty_categories': difficulty_categories,
    'qv_metrics': {str(k): v for k, v in qv_metrics.items()},
    'confidence_analysis': {k: v for k, v in confidence_analysis.items()},
    'improvement_targets': {
        'medium_confidence_positions': [(int(qv), float(pct), float(recall)) for qv, pct, recall in medium_confidence_positions[:20]],
        'high_fp_positions': [(int(qv), float(fp), float(recall)) for qv, fp, recall in high_fp_positions[:20]],
    },
    'recommendations': recommendations,
}

summary_path = output_dir / "comprehensive_summary.json"
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print(f"  [OK] Summary saved: {summary_path}")

# Save detailed event-by-event results (JSON)
events_path = output_dir / "event_by_event_results.json"
with open(events_path, 'w', encoding='utf-8') as f:
    json.dump(event_details, f, indent=2)
print(f"  [OK] Event details saved: {events_path}")

# Save markdown report
report_path = output_dir / "diagnostic_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Comprehensive 1000-Event Holdout Diagnostic Report\n\n")
    f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Model**: lgbm_ranker_baseline_fixed_v1.pkl (Fixed Architecture - NO LEAKAGE)\n")
    f.write(f"**Events Tested**: {total_events}\n\n")

    f.write(f"## HOLDOUT TEST SUMMARY - {total_events} Events\n")
    f.write("```\n")
    f.write("  ------------------------------------\n")
    for num_wrong in range(6):
        count = wrong_counts.get(num_wrong, 0)
        pct = (count / total_events) * 100
        label = {0: '← All 5 actual values in top-20  ✓ EXCELLENT',
                 1: '← 4 of 5 actual values in top-20  ✓ GOOD',
                 2: '← 3 of 5 actual values in top-20  ✗ POOR',
                 3: '← 2 of 5 actual values in top-20  ✗ POOR',
                 4: '← 1 of 5 actual values in top-20  ~ ACCEPTABLE',
                 5: '← 0 of 5 actual values in top-20  ~ ACCEPTABLE'}
        f.write(f"  {num_wrong} wrong: {count:4d} events ({pct:5.2f}%)  {label[num_wrong]}\n")
    f.write("  ------------------------------------\n")
    f.write(f"\nOverall Recall@20: {overall_recall:.2%} ({total_correct}/{total_positions} correct)\n")
    f.write("```\n\n")

    f.write("## MACRO OBJECTIVE ANALYSIS\n\n")
    f.write(f"**Goal**: Force results to EXTREMES (0-1 wrong OR 4-5 wrong)\n\n")
    f.write(f"- **Excellent/Good (0-1 wrong)**: {excellent_good_pct:.1f}% ← WANT HIGHER\n")
    f.write(f"- **Poor (2-3 wrong)**: {poor_pct:.1f}% ← WANT LOWER (stuck in middle!)\n")
    f.write(f"- **Acceptable (4-5 wrong)**: {acceptable_pct:.1f}% ← OK\n\n")

    f.write("## TOP IMPROVEMENT TARGETS\n\n")
    f.write("### Positions Stuck in Medium Confidence (0.4-0.6)\n")
    f.write("These positions need stronger signals to push to extremes:\n\n")
    for qv_pos, medium_pct, recall in medium_confidence_positions[:15]:
        f.write(f"- **QV {qv_pos}**: {medium_pct:.1%} in medium zone, recall={recall:.1%}\n")

    f.write("\n### High False Positive Positions\n")
    f.write("These are predicted too often when wrong:\n\n")
    for qv_pos, fp_rate, recall in high_fp_positions[:10]:
        f.write(f"- **QV {qv_pos}**: FP rate={fp_rate:.1%}, recall={recall:.1%}\n")

    f.write("\n## ACTIONABLE RECOMMENDATIONS\n\n")
    for category, recs in recommendations.items():
        if recs:
            f.write(f"### {category.replace('_', ' ').title()}\n\n")
            for rec in recs:
                f.write(f"#### {rec['target']} [{rec['priority']}]\n\n")
                f.write(f"**Problem**: {rec['problem']}\n\n")
                f.write(f"**Solution**: {rec['solution']}\n\n")
                f.write(f"**Expected Impact**: {rec['expected_impact']}\n\n")
                f.write(f"**Implementation**: {rec['implementation']}\n\n")

print(f"  [OK] Markdown report saved: {report_path}")

# Print final summary
print(f"\n" + "=" * 80)
print(f"COMPREHENSIVE {total_events}-EVENT DIAGNOSTIC COMPLETE")
print("=" * 80)
print(f"\nHOLDOUT TEST SUMMARY - {total_events} Events")
print(f"Model: lgbm_ranker_baseline_fixed_v1.pkl")
print(f"  ------------------------------------")
for num_wrong in range(6):
    count = wrong_counts.get(num_wrong, 0)
    pct = (count / total_events) * 100
    label = {0: '← All 5 actual values in top-20  ✓ EXCELLENT',
             1: '← 4 of 5 actual values in top-20  ✓ GOOD',
             2: '← 3 of 5 actual values in top-20  ✗ POOR',
             3: '← 2 of 5 actual values in top-20  ✗ POOR',
             4: '← 1 of 5 actual values in top-20  ~ ACCEPTABLE',
             5: '← 0 of 5 actual values in top-20  ~ ACCEPTABLE'}
    print(f"  {num_wrong} wrong: {count:4d} events ({pct:5.2f}%)  {label[num_wrong]}")
print(f"  ------------------------------------")
print(f"\nOverall Recall@20: {overall_recall:.2%} ({total_correct}/{total_positions} correct)")

print(f"\n📊 KEY FINDINGS:")
print(f"  • {poor_pct:.1f}% stuck in middle (2-3 wrong) ← MAIN PROBLEM")
print(f"  • {len(medium_confidence_positions)} positions in medium confidence zone")
print(f"  • {medium_zone_pct:.1f}% of all predictions have wishy-washy confidence")

print(f"\n🎯 TOP RECOMMENDATIONS:")
for category in ['algorithm_modifications', 'feature_enhancements', 'imputation_improvements']:
    high_priority = [r for r in recommendations[category] if r['priority'] == 'HIGH']
    if high_priority:
        print(f"  • {high_priority[0]['target']}: {high_priority[0]['solution']}")

print(f"\n📁 Full results saved to: {output_dir}")
print(f"  • comprehensive_summary.json (all metrics)")
print(f"  • event_by_event_results.json (detailed per-event data)")
print(f"  • diagnostic_report.md (human-readable report)")

print(f"\n✅ Ready for improvement implementation!")
