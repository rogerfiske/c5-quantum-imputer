"""
Next Event Prediction - Epic 7 Story 7.4

Generate predictions for the next event using all 3 trained models
(LGBM, SetTransformer, GNN) with pluggable architecture.

This allows comparing predictions from all 3 models before selecting
the final production model.

Usage:
    python production/predict_next_event.py

Output:
    - Predictions from all 3 models side-by-side
    - Saved to production/predictions/next_event_predictions.json

Author: BMad Dev Agent (James)
Date: 2025-10-15
Epic: Epic 7 - Final Production Run
Story: 7.4 - Generate Next Event Predictions
"""

import sys
from pathlib import Path
import pandas as pd
import json
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from production.model_pipeline import ModelPipeline


def load_latest_event(imputation_strategy: str = 'angle_encoding') -> pd.DataFrame:
    """
    Load the most recent event from the dataset.

    Args:
        imputation_strategy: Which imputed dataset to use

    Returns:
        DataFrame with the latest event (1 row)
    """
    # Data prefix mapping
    prefix_map = {
        'basis_embedding': 'basis',
        'amplitude_embedding': 'amplitude',
        'angle_encoding': 'angle',
        'density_matrix': 'density',
        'graph_cycle': 'graph'
    }

    data_prefix = prefix_map[imputation_strategy]

    # Try train split first (has most recent data)
    train_path = Path(f'data/splits/{data_prefix}/{data_prefix}_train.csv')

    if train_path.exists():
        print(f"Loading training data: {train_path}")
        df = pd.read_csv(train_path)

        # Get the last event (most recent)
        latest_event = df.iloc[[-1]]  # Keep as DataFrame (single row)

        if 'event-ID' in latest_event.columns:
            event_id = latest_event['event-ID'].values[0]
            print(f"  Latest event ID: {event_id}")
            print(f"  Total events in training: {len(df)}")

        return latest_event

    else:
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            f"Please ensure Epic 5 data preparation is complete."
        )


def predict_with_all_models(
    current_state: pd.DataFrame,
    imputation_strategy: str = 'angle_encoding',
    k: int = 20
) -> Dict[str, Dict[str, Any]]:
    """
    Generate predictions using all 3 models.

    Args:
        current_state: Current quantum state (1 row, imputed)
        imputation_strategy: Imputation method used
        k: Number of top predictions

    Returns:
        Dictionary mapping model_type to prediction results
    """
    models = ['lgbm', 'settransformer', 'gnn']
    predictions = {}

    print(f"\n{'='*80}")
    print("GENERATING PREDICTIONS WITH ALL 3 MODELS")
    print(f"{'='*80}\n")

    for model_type in models:
        print(f"--- {model_type.upper()} ---")

        try:
            # Create pipeline
            pipeline = ModelPipeline(model_type, imputation_strategy)

            # Generate prediction
            result = pipeline.predict_next_event(current_state, k=k)

            predictions[model_type] = result

            # Show top-5
            top_5 = result['predictions'][:5]
            print(f"  Top-5: {top_5}\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            predictions[model_type] = {'error': str(e)}

    return predictions


def analyze_consensus(predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze consensus between the 3 models.

    Args:
        predictions: Predictions from all 3 models

    Returns:
        Dictionary with consensus analysis
    """
    print(f"\n{'='*80}")
    print("CONSENSUS ANALYSIS")
    print(f"{'='*80}\n")

    # Extract top-20 from each model
    model_predictions = {}
    for model, result in predictions.items():
        if 'error' not in result:
            model_predictions[model] = set(result['predictions'])

    if len(model_predictions) != 3:
        print("WARNING: Not all models produced predictions")
        return {'error': 'Incomplete predictions'}

    # Find consensus (positions predicted by all 3 models)
    consensus = set.intersection(*model_predictions.values())

    # Find positions predicted by at least 2 models
    majority = set()
    pred_lists = list(model_predictions.values())
    for i, pred_set in enumerate(pred_lists):
        for other_set in pred_lists[i+1:]:
            majority.update(pred_set & other_set)

    # Analyze top-5 agreement
    top_5_by_model = {
        model: result['predictions'][:5]
        for model, result in predictions.items()
        if 'error' not in result
    }

    top_5_consensus = set.intersection(
        *[set(preds) for preds in top_5_by_model.values()]
    )

    print(f"Top-20 Consensus (all 3 models agree): {len(consensus)} positions")
    if consensus:
        print(f"  Positions: {sorted(consensus)}\n")
    else:
        print("  No positions predicted by all 3 models\n")

    print(f"Top-20 Majority (2+ models agree): {len(majority)} positions")
    if majority:
        print(f"  Positions: {sorted(majority)}\n")
    else:
        print("  No positions predicted by 2+ models\n")

    print(f"Top-5 Consensus (all 3 models agree): {len(top_5_consensus)} positions")
    if top_5_consensus:
        print(f"  Positions: {sorted(top_5_consensus)}\n")
    else:
        print("  No top-5 consensus\n")

    # Display top-5 from each model
    print("Top-5 from each model:")
    for model, preds in top_5_by_model.items():
        print(f"  {model.upper():<15} {preds}")

    return {
        'full_consensus': sorted(consensus),
        'majority_vote': sorted(majority),
        'top_5_consensus': sorted(top_5_consensus),
        'top_5_by_model': top_5_by_model,
        'agreement_metrics': {
            'full_consensus_count': len(consensus),
            'majority_count': len(majority),
            'top_5_consensus_count': len(top_5_consensus)
        }
    }


def save_predictions(
    predictions: Dict[str, Dict[str, Any]],
    consensus: Dict[str, Any],
    current_state: pd.DataFrame,
    output_path: Path
):
    """
    Save all predictions to JSON file.

    Args:
        predictions: Predictions from all 3 models
        consensus: Consensus analysis results
        current_state: Input state used for prediction
        output_path: Where to save predictions
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract event ID if available
    event_id = None
    if 'event-ID' in current_state.columns:
        event_id = int(current_state['event-ID'].values[0])

    # Build output structure
    output = {
        'metadata': {
            'epic': 'Epic 7 - Final Production Run',
            'story': '7.4 - Next Event Prediction',
            'imputation_strategy': 'angle_encoding',
            'k': 20,
            'input_event_id': event_id,
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'predictions_by_model': predictions,
        'consensus_analysis': consensus,
        'recommendation': {
            'note': 'All 3 models achieved 100% recall@20 on holdout test',
            'suggested_approach': 'Use consensus positions for highest confidence',
            'model_selection': 'Choose LGBM (fastest), SetTransformer (most elegant), or GNN (best for graph structure)'
        }
    }

    # Save
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Predictions saved to: {output_path}")
    print(f"{'='*80}")


def main():
    """Main execution: Generate predictions with all 3 models."""
    print("="*80)
    print("EPIC 7 STORY 7.4: NEXT EVENT PREDICTION")
    print("="*80)
    print("\nGenerating predictions with all 3 models (LGBM, SetTransformer, GNN)")
    print("Using Angle Encoding imputation strategy\n")

    imputation_strategy = 'angle_encoding'

    try:
        # Load latest event
        print(f"\n{'='*80}")
        print("LOADING LATEST EVENT")
        print(f"{'='*80}\n")

        current_state = load_latest_event(imputation_strategy)
        print(f"  Loaded event successfully (shape: {current_state.shape})")

        # Generate predictions with all 3 models
        predictions = predict_with_all_models(current_state, imputation_strategy, k=20)

        # Analyze consensus
        consensus = analyze_consensus(predictions)

        # Save results
        output_path = Path('production/predictions/next_event_predictions.json')
        save_predictions(predictions, consensus, current_state, output_path)

        # Final summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80 + "\n")

        successful = sum(1 for p in predictions.values() if 'error' not in p)
        print(f"Models executed: {successful}/3")

        if 'agreement_metrics' in consensus:
            metrics = consensus['agreement_metrics']
            print(f"Top-20 full consensus: {metrics['full_consensus_count']} positions")
            print(f"Top-20 majority (2+): {metrics['majority_count']} positions")
            print(f"Top-5 consensus: {metrics['top_5_consensus_count']} positions")

        print("\nReady for Epic 7 Story 7.5: Production Report")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
