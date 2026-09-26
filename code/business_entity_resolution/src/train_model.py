import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
import argparse
import random

def generate_synthetic_positives(s1_df: pd.DataFrame, n_pos: int = 500) -> tuple:
    """
    Generate realistic synthetic positive pairs by injecting standard entity resolution noise
    (typos, abbreviations, word swaps, dropped address components).
    """
    sample_s1 = s1_df.sample(min(n_pos, len(s1_df)), random_state=42).copy()
    
    noisy_cands = []
    pairs = []
    
    for idx, row in sample_s1.iterrows():
        s1_id = row['entity_id']
        cand_id = f"SYN-{s1_id}"
        
        name = str(row.get('norm_name', ''))
        addr = str(row.get('norm_address', ''))
        country = str(row.get('country', ''))
        
        # Inject name noise
        tokens = name.split()
        if len(tokens) > 1 and random.random() < 0.4:
            # Word swap
            tokens[0], tokens[-1] = tokens[-1], tokens[0]
            name = ' '.join(tokens)
        elif len(name) > 5 and random.random() < 0.5:
            # Typo (delete or replace a character)
            pos = random.randint(1, len(name) - 2)
            name = name[:pos] + name[pos+1:]
            
        # Inject address noise
        if len(addr) > 8 and random.random() < 0.5:
            addr_tokens = addr.split()
            if len(addr_tokens) > 2:
                addr = ' '.join(addr_tokens[:-1]) # drop last token (like postal code)
                
        noisy_cands.append({
            'entity_id': cand_id,
            'norm_name': name,
            'norm_address': addr,
            'country': country
        })
        pairs.append((s1_id, cand_id))
        
    cand_df_synth = pd.DataFrame(noisy_cands)
    pos_pairs = pd.DataFrame(pairs, columns=['s1_id', 'cand_id'])
    pos_pairs['label'] = 1
    
    return pos_pairs, cand_df_synth

def build_training_data(s1_df: pd.DataFrame, cand_df: pd.DataFrame, gt_df: pd.DataFrame, source_name: str, neg_ratio: int = 5):
    from blocking import get_candidate_pairs
    from features import generate_features
    
    print(f"[{source_name}] Generating candidate pairs via blocking...")
    pairs = get_candidate_pairs(s1_df, cand_df, source_name)
    
    print(f"[{source_name}] Labeling candidates using ground truth...")
    gt = gt_df.copy()
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('').str.split(',')
    gt_exploded = gt.explode('matched_entity_ids')
    gt_exploded = gt_exploded[gt_exploded['matched_entity_ids'] != '']
    gt_exploded.columns = ['s1_id', 'cand_id']
    gt_exploded['label'] = 1
    
    labeled_pairs = pairs.merge(gt_exploded, on=['s1_id', 'cand_id'], how='left')
    labeled_pairs['label'] = labeled_pairs['label'].fillna(0).astype(int)
    
    positives = labeled_pairs[labeled_pairs['label'] == 1]
    negatives = labeled_pairs[labeled_pairs['label'] == 0]
    
    print(f"[{source_name}] Positive pairs found in GT: {len(positives)}, Negative pairs: {len(negatives)}")
    
    # Fallback: if Ground Truth is disjoint from the sample slices, synthesize realistic positives
    cand_df_for_features = cand_df
    if len(positives) == 0:
        print(f"[{source_name}] Notice: Ground truth has 0 overlap with {source_name} sample. Generating synthetic positive pairs...")
        synth_pos, synth_cand_df = generate_synthetic_positives(s1_df, n_pos=min(300, len(s1_df)))
        positives = synth_pos
        cand_df_for_features = pd.concat([cand_df, synth_cand_df], ignore_index=True)
        
    n_neg = min(len(negatives), len(positives) * neg_ratio)
    if n_neg > 0:
        negatives_sampled = negatives.sample(n=n_neg, random_state=42)
    else:
        negatives_sampled = negatives
        
    training_pairs = pd.concat([positives, negatives_sampled], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[{source_name}] Generating features for {len(training_pairs)} training pairs...")
    features_df = generate_features(training_pairs[['s1_id', 'cand_id']], s1_df, cand_df_for_features)
    
    training_data = features_df.merge(training_pairs[['s1_id', 'cand_id', 'label']], on=['s1_id', 'cand_id'])
    return training_data

def train_model(train_data: pd.DataFrame, model_path: str):
    feature_cols = [c for c in train_data.columns if c not in ['s1_id', 'cand_id', 'label']]
    print(f"Training LightGBM on {len(train_data)} rows. Features: {feature_cols}")
    
    X = train_data[feature_cols]
    y = train_data['label']
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 10,
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', default='dataset_sample/train/train_source1.tsv')
    parser.add_argument('--s2', default='dataset_sample/train/train_source2.tsv')
    parser.add_argument('--s3', default='dataset_sample/train/train_source3.tsv')
    parser.add_argument('--gt', default='dataset_sample/train/train_ground_truth.tsv')
    parser.add_argument('--model', default='model/lgbm_model.pkl')
    args = parser.parse_args()
    
    # Path fallbacks for full dataset if present
    s1_path = args.s1 if os.path.exists(args.s1) else 'dataset/train/train_source1.tsv'
    s2_path = args.s2 if os.path.exists(args.s2) else 'dataset/train/train_source2.tsv'
    s3_path = args.s3 if os.path.exists(args.s3) else 'dataset/train/train_source3.tsv'
    gt_path = args.gt if os.path.exists(args.gt) else 'dataset/train/train_ground_truth.tsv'
    
    print(f"Loading data: S1={s1_path}, S2={s2_path}, S3={s3_path}, GT={gt_path}")
    s1 = pd.read_csv(s1_path, sep='\t')
    s2 = pd.read_csv(s2_path, sep='\t')
    s3 = pd.read_csv(s3_path, sep='\t')
    gt = pd.read_csv(gt_path, sep='\t')
    
    print("Normalizing training data...")
    s1_norm = process_dataframe(s1)
    s2_norm = process_dataframe(s2)
    s3_norm = process_dataframe(s3)
    
    print("Building training data for S2 and S3...")
    data_s2 = build_training_data(s1_norm, s2_norm, gt, "S2", neg_ratio=5)
    data_s3 = build_training_data(s1_norm, s3_norm, gt, "S3", neg_ratio=5)
    
    full_train = pd.concat([data_s2, data_s3], ignore_index=True)
    
    if len(full_train) > 0:
        train_model(full_train, args.model)
    else:
        print("No training data generated.")
