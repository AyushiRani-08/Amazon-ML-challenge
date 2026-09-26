import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os

def build_training_data(s1_df: pd.DataFrame, cand_df: pd.DataFrame, gt_df: pd.DataFrame, source_name: str, neg_ratio: int = 5):
    """
    Builds training data by generating candidate pairs (via blocking), labeling them using ground truth,
    and computing features.
    Samples negatives to maintain a pos:neg ratio.
    """
    from blocking import get_candidate_pairs
    from features import generate_features
    
    print(f"[{source_name}] Generating candidate pairs...")
    pairs = get_candidate_pairs(s1_df, cand_df, source_name)
    
    print(f"[{source_name}] Labeling candidates using ground truth...")
    # Explode ground truth
    gt = gt_df.copy()
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('').str.split(',')
    gt_exploded = gt.explode('matched_entity_ids')
    gt_exploded = gt_exploded[gt_exploded['matched_entity_ids'] != '']
    gt_exploded.columns = ['s1_id', 'cand_id']
    gt_exploded['label'] = 1
    
    # Merge with pairs
    labeled_pairs = pairs.merge(gt_exploded, on=['s1_id', 'cand_id'], how='left')
    labeled_pairs['label'] = labeled_pairs['label'].fillna(0).astype(int)
    
    # Wait, what if true matches were not retrieved by blocking?
    # To help the model learn, we should always include all true matches in training (even if missed by blocking).
    # Since in inference they won't be retrieved, the model won't see them, but it helps learn the difference better.
    missed_positives = gt_exploded.merge(pairs, on=['s1_id', 'cand_id'], how='left', indicator=True)
    missed_positives = missed_positives[missed_positives['_merge'] == 'left_only'][['s1_id', 'cand_id', 'label']]
    
    # Filter missed positives to only include candidates that are in cand_df
    # (Since gt might contain both S2 and S3, and cand_df is only one of them)
    missed_positives = missed_positives[missed_positives['cand_id'].isin(cand_df['entity_id'])]
    
    labeled_pairs = pd.concat([labeled_pairs, missed_positives], ignore_index=True).drop_duplicates(subset=['s1_id', 'cand_id'])
    
    print(f"[{source_name}] Positive pairs: {labeled_pairs['label'].sum()}, Negative pairs: {(labeled_pairs['label'] == 0).sum()}")
    
    # Subsample negatives
    positives = labeled_pairs[labeled_pairs['label'] == 1]
    negatives = labeled_pairs[labeled_pairs['label'] == 0]
    
    n_neg = min(len(negatives), len(positives) * neg_ratio)
    if n_neg > 0:
        negatives_sampled = negatives.sample(n=n_neg, random_state=42)
    else:
        negatives_sampled = negatives
        
    training_pairs = pd.concat([positives, negatives_sampled], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[{source_name}] Generating features for {len(training_pairs)} pairs...")
    features_df = generate_features(training_pairs[['s1_id', 'cand_id']], s1_df, cand_df)
    
    # Join label back
    training_data = features_df.merge(training_pairs[['s1_id', 'cand_id', 'label']], on=['s1_id', 'cand_id'])
    
    return training_data

def train_model(train_data: pd.DataFrame, model_path: str):
    """
    Train LightGBM binary classifier on the computed features.
    """
    feature_cols = [c for c in train_data.columns if c not in ['s1_id', 'cand_id', 'label']]
    print(f"Training on {len(train_data)} rows. Features: {feature_cols}")
    
    X = train_data[feature_cols]
    y = train_data['label']
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'random_state': 42,
        'verbose': -1
    }
    
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=100)
    
    print("Model trained. Feature importance:")
    importance = model.feature_importance()
    for feat, imp in zip(feature_cols, importance):
        print(f"  {feat}: {imp}")
        
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    return model

if __name__ == "__main__":
    from normalize import process_dataframe
    
    print("Loading data...")
    # Read larger sample for training
    s1 = pd.read_csv('dataset_sample/train/train_source1.tsv', sep='\t')
    s2 = pd.read_csv('dataset_sample/train/train_source2.tsv', sep='\t')
    gt = pd.read_csv('dataset_sample/train/train_ground_truth.tsv', sep='\t')
    
    print("Normalizing...")
    s1_norm = process_dataframe(s1)
    s2_norm = process_dataframe(s2)
    
    print("Building training data...")
    # Just train on S2 for the test script
    train_data = build_training_data(s1_norm, s2_norm, gt, "S2", neg_ratio=5)
    
    if len(train_data) > 0:
        train_model(train_data, 'model/lgbm_model.pkl')
    else:
        print("No training data generated.")
