import pandas as pd
import numpy as np
import os
import subprocess
from normalize import process_dataframe
from blocking import get_candidate_pairs
from features import generate_features
from threshold_and_predict import predict, resolve_global_consistency

def generate_outputs(test_s1_path: str, test_s2_path: str, test_s3_path: str, model_path: str, output_dir: str, threshold: float = 0.5):
    print("Loading test data...")
    s1 = pd.read_csv(test_s1_path, sep='\t')
    s2 = pd.read_csv(test_s2_path, sep='\t')
    s3 = pd.read_csv(test_s3_path, sep='\t')
    
    print("Normalizing...")
    s1_norm = process_dataframe(s1)
    s2_norm = process_dataframe(s2)
    s3_norm = process_dataframe(s3)
    
    print("Generating candidate pairs...")
    pairs_s2 = get_candidate_pairs(s1_norm, s2_norm, "S2")
    pairs_s3 = get_candidate_pairs(s1_norm, s3_norm, "S3")
    all_pairs = pd.concat([pairs_s2, pairs_s3], ignore_index=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write candidate_pairs.tsv
    print("Writing candidate_pairs.tsv...")
    grouped_cands = all_pairs.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(x)).reset_index()
    grouped_cands.columns = ['source1_entity_id', 'candidate_entity_ids']
    
    # Ensure all S1 entities are present
    base_s1 = s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
    candidate_pairs_output = base_s1.merge(grouped_cands, on='source1_entity_id', how='left')
    candidate_pairs_output['candidate_entity_ids'] = candidate_pairs_output['candidate_entity_ids'].fillna('')
    
    candidate_pairs_output.to_csv(os.path.join(output_dir, 'candidate_pairs.tsv'), sep='\t', index=False)
    
    print("Generating features for candidates...")
    # Features for S2
    feats_s2 = generate_features(pairs_s2, s1_norm, s2_norm)
    # Features for S3
    feats_s3 = generate_features(pairs_s3, s1_norm, s3_norm)
    
    all_feats = pd.concat([feats_s2, feats_s3], ignore_index=True)
    
    print("Predicting scores...")
    predictions = predict(model_path, all_feats)
    
    print("Resolving global consistency...")
    resolved = resolve_global_consistency(predictions, threshold)
    
    # Write matching_results.tsv
    print("Writing matching_results.tsv...")
    matching_results = base_s1.merge(resolved, on='source1_entity_id', how='left')
    matching_results['matched_entity_ids'] = matching_results['matched_entity_ids'].fillna('')
    
    matching_results.to_csv(os.path.join(output_dir, 'matching_results.tsv'), sep='\t', index=False)
    print("Output generation complete.")
    
    # Run validation
    val_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils/validate_submission.py'))
    if not os.path.exists(val_script):
        val_script = 'utils/validate_submission.py'
        
    if os.path.exists(val_script):
        print("Running validation script...")
        import sys
        result = subprocess.run([
            sys.executable, val_script,
            '--matching', os.path.join(output_dir, 'matching_results.tsv'),
            '--candidate', os.path.join(output_dir, 'candidate_pairs.tsv'),
            '--test-dir', os.path.dirname(test_s1_path)
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print("Validation PASS")
        else:
            print("Validation FAIL:")
            print(result.stdout)
            print(result.stderr)
    else:
        print(f"Validation script not found at {val_script}. Skipping.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--s1', default='../../dataset/test/test_source1.tsv')
    parser.add_argument('--s2', default='../../dataset/test/test_source2.tsv')
    parser.add_argument('--s3', default='../../dataset/test/test_source3.tsv')
    parser.add_argument('--model', default='../../model/lgbm_model.pkl')
    parser.add_argument('--out', default='../../output')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # If the paths don't exist, we fallback to sample for testing
    s1_path = args.s1 if os.path.exists(args.s1) else 'dataset_sample/test/test_source1.tsv'
    s2_path = args.s2 if os.path.exists(args.s2) else 'dataset_sample/test/test_source2.tsv'
    s3_path = args.s3 if os.path.exists(args.s3) else 'dataset_sample/test/test_source3.tsv'
    model_path = args.model if os.path.exists(args.model) else 'model/lgbm_model.pkl'
    out_dir = args.out if os.path.exists(args.s1) else 'output'
    
    if os.path.exists('dataset_sample/test/test_source1.tsv'):
        generate_outputs(s1_path, s2_path, s3_path, model_path, out_dir, args.threshold)
    else:
        print("Test data not found.")
