import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict
from evaluate import compute_f05_score

def resolve_global_consistency(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Apply global consistency resolution: if the same S2/S3 ID is proposed as best match 
    for multiple S1 entities, resolve via confidence-based greedy assignment.
    """
    # Filter by threshold first
    valid_preds = predictions[predictions['score'] >= threshold].copy()
    
    # Sort by score descending for greedy assignment
    valid_preds = valid_preds.sort_values(by='score', ascending=False)
    
    assigned_cands = set()
    final_matches = []
    
    # For 1M rows, iterrows might be slow, so we can use drop_duplicates on cand_id
    # Since it's sorted by score descending, dropping duplicates on cand_id keeps the highest scoring S1 for each candidate
    # Wait, S1 can match MULTIPLE candidates (e.g. one S2, one S3, or multiple S2 if duplicates).
    # The requirement: "if the same S2/S3 ID is proposed as best match for multiple S1 entities, resolve..."
    # This implies a candidate can only belong to ONE S1 entity.
    
    resolved_preds = valid_preds.drop_duplicates(subset=['cand_id'])
    
    # Now group by S1 to get the final list of candidates
    grouped = resolved_preds.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(x)).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    return grouped

def sweep_threshold(predictions: pd.DataFrame, ground_truth: pd.DataFrame, s1_ids_df: pd.DataFrame) -> float:
    """
    Sweep decision threshold on a validation set to maximize F_0.5.
    predictions contains: s1_id, cand_id, score
    s1_ids_df contains all S1 IDs in the validation set (needed to include singletons).
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f05 = -1.0
    best_thresh = 0.5
    
    base_s1 = s1_ids_df[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
    
    for t in thresholds:
        resolved = resolve_global_consistency(predictions, t)
        
        # Merge with all S1 IDs to handle singletons (empty matches)
        full_preds = base_s1.merge(resolved, on='source1_entity_id', how='left')
        full_preds['matched_entity_ids'] = full_preds['matched_entity_ids'].fillna('')
        
        f05 = compute_f05_score(ground_truth, full_preds)
        print(f"Threshold: {t:.2f} -> F0.5: {f05:.4f}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
            
    print(f"Best Threshold: {best_thresh:.2f} with F0.5: {best_f05:.4f}")
    return best_thresh

def predict(model_path: str, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict scores using the trained model.
    """
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    feature_cols = [c for c in features_df.columns if c not in ['s1_id', 'cand_id', 'label']]
    X = features_df[feature_cols]
    
    scores = model.predict(X)
    
    result = features_df[['s1_id', 'cand_id']].copy()
    result['score'] = scores
    
    return result

if __name__ == "__main__":
    pass
