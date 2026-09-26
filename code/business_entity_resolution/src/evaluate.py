import pandas as pd
import numpy as np

def compute_f05_score(ground_truth_df: pd.DataFrame, prediction_df: pd.DataFrame) -> float:
    """
    Computes macro-averaged per-entity F_0.5 score.
    ground_truth_df: columns ['source1_entity_id', 'matched_entity_ids']
    prediction_df: columns ['source1_entity_id', 'matched_entity_ids']
    """
    
    # Merge predictions with ground truth
    df = ground_truth_df.merge(prediction_df, on='source1_entity_id', how='left', suffixes=('_gt', '_pred'))
    df['matched_entity_ids_pred'] = df['matched_entity_ids_pred'].fillna('')
    df['matched_entity_ids_gt'] = df['matched_entity_ids_gt'].fillna('')
    
    f05_scores = []
    
    for _, row in df.iterrows():
        gt_str = row['matched_entity_ids_gt'].strip()
        pred_str = row['matched_entity_ids_pred'].strip()
        
        gt_set = set(gt_str.split(',')) if gt_str else set()
        pred_set = set(pred_str.split(',')) if pred_str else set()
        
        # Handle empty sets (singletons)
        if not gt_set and not pred_set:
            f05_scores.append(1.0)
            continue
        elif not gt_set or not pred_set:
            f05_scores.append(0.0)
            continue
            
        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        if tp == 0:
            f05_scores.append(0.0)
            continue
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        f05 = 1.25 * (precision * recall) / (0.25 * precision + recall)
        f05_scores.append(f05)
        
    macro_f05 = np.mean(f05_scores)
    return macro_f05

if __name__ == "__main__":
    # Test evaluation metric
    gt = pd.DataFrame({
        'source1_entity_id': ['S1-1', 'S1-2', 'S1-3'],
        'matched_entity_ids': ['S2-1,S3-1', '', 'S2-2']
    })
    
    pred = pd.DataFrame({
        'source1_entity_id': ['S1-1', 'S1-2', 'S1-3'],
        'matched_entity_ids': ['S2-1', '', 'S2-3']
    })
    
    score = compute_f05_score(gt, pred)
    print(f"F0.5 Score: {score}")
