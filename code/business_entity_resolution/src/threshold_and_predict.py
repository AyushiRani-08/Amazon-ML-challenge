import pandas as pd
import numpy as np
import pickle
from typing import Dict
from evaluate import compute_f05_score

def resolve_global_consistency(predictions: pd.DataFrame, threshold: float = 0.65) -> pd.DataFrame:
    """
    Apply high-precision filtering and global consistency resolution:
    1. Score >= threshold (calibrated for F_0.5 precision weighting)
    2. Country must match (country_match == 1)
    3. Name similarity guardrail (name_jaro_winkler >= 0.70) to protect singletons from false merges
    4. 1-to-1 candidate assignment: an S2/S3 candidate entity can only be assigned to one S1 entity
    """
    if len(predictions) == 0:
        return pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
        
    valid = predictions[predictions['score'] >= threshold].copy()
    
    # Singleton protection guardrails
    if 'country_match' in valid.columns:
        valid = valid[valid['country_match'] == 1]
    if 'name_jaro_winkler' in valid.columns:
        valid = valid[valid['name_jaro_winkler'] >= 0.70]
        
    if len(valid) == 0:
        return pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
        
    # Sort by score descending for greedy confidence assignment
    valid_preds = valid.sort_values(by='score', ascending=False)
    
    # 1-to-1 constraint: each candidate belongs to at most one S1
    resolved_preds = valid_preds.drop_duplicates(subset=['cand_id'])
    
    # Group by S1 to compile matched candidate IDs
    grouped = resolved_preds.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(x)).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    return grouped

def sweep_threshold(predictions: pd.DataFrame, ground_truth: pd.DataFrame, s1_ids_df: pd.DataFrame) -> float:
    """
    Sweep decision threshold on a validation set to directly maximize macro F_0.5.
    """
    thresholds = np.linspace(0.4, 0.85, 10)
    best_f05 = -1.0
    best_thresh = 0.65
    
    base_s1 = s1_ids_df[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
    
    for t in thresholds:
        resolved = resolve_global_consistency(predictions, t)
        
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
    Predict match probability scores using an ensemble of the trained model
    and calibrated string alignment evidence.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    feature_cols = [c for c in features_df.columns if c not in ['s1_id', 'cand_id', 'label']]
    model_scores = model.predict(features_df[feature_cols])
    
    result = features_df[['s1_id', 'cand_id']].copy()
    
    for col in ['name_jaro_winkler', 'name_tfidf_cosine', 'addr_jaro_winkler', 'addr_tfidf_cosine', 'country_match']:
        if col in features_df.columns:
            result[col] = features_df[col]
            
    # Hybrid calibrated composite score
    country = result['country_match'] if 'country_match' in result.columns else 1
    name_jw = result['name_jaro_winkler'] if 'name_jaro_winkler' in result.columns else 0
    name_tfidf = result['name_tfidf_cosine'] if 'name_tfidf_cosine' in result.columns else 0
    addr_jw = result['addr_jaro_winkler'] if 'addr_jaro_winkler' in result.columns else 0
    
    composite = (
        0.35 * name_jw +
        0.25 * name_tfidf +
        0.20 * addr_jw +
        0.20 * model_scores
    ) * country
    
    high_conf = (model_scores >= 0.50) | (
        (name_jw >= 0.88) & 
        (name_tfidf >= 0.65) &
        (addr_jw >= 0.35)
    )
    
    result['score'] = np.where(high_conf, np.maximum(composite, model_scores), composite * 0.4)
    return result

if __name__ == "__main__":
    pass
