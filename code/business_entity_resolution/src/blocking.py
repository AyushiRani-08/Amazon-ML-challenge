import pandas as pd
import numpy as np
import jellyfish
from typing import List, Set
import time
from normalize import process_dataframe

def build_phonetic_blocks(df: pd.Series) -> pd.Series:
    """
    Generate Soundex codes for phonetic blocking.
    """
    # handle NaN and non-strings
    s = df.fillna('').astype(str)
    # jellyfish.soundex is fast
    return s.apply(lambda x: jellyfish.soundex(x) if x else '')

def build_token_blocks(df: pd.Series, max_tokens: int = 3) -> pd.DataFrame:
    """
    Explode top N longest tokens for token-overlap blocking.
    """
    s = df.fillna('').astype(str)
    
    def get_longest_tokens(text):
        tokens = text.split()
        if not tokens:
            return []
        # Sort by length descending, take top max_tokens
        tokens.sort(key=len, reverse=True)
        return tokens[:max_tokens]
        
    exploded = s.apply(get_longest_tokens).explode()
    # return a dataframe with the original index and the token
    return exploded[exploded.str.len() > 3] # only consider tokens length > 3 to avoid stop words matching too much

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def build_tfidf_knn_pairs(s1_df: pd.DataFrame, cand_df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """
    Generate candidate pairs using character n-gram TF-IDF Nearest Neighbors.
    Captures typos, abbreviations, and word order variations with high recall.
    """
    s1_text = s1_df['norm_name'].fillna('') + ' ' + s1_df['norm_address'].fillna('')
    cand_text = cand_df['norm_name'].fillna('') + ' ' + cand_df['norm_address'].fillna('')
    
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1, max_features=40000)
    X_cand = vec.fit_transform(cand_text)
    X_s1 = vec.transform(s1_text)
    
    k = min(top_k, cand_df.shape[0])
    if k == 0:
        return pd.DataFrame(columns=['s1_id', 'cand_id'])
        
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(X_cand)
    distances, indices = nn.kneighbors(X_s1)
    
    s1_ids = s1_df['entity_id'].values
    cand_ids = cand_df['entity_id'].values
    
    rows = []
    for i in range(len(s1_ids)):
        curr_s1 = s1_ids[i]
        for dist, idx in zip(distances[i], indices[i]):
            if dist < 0.85:
                rows.append((curr_s1, cand_ids[idx]))
                
    return pd.DataFrame(rows, columns=['s1_id', 'cand_id'])

def get_candidate_pairs(s1_df: pd.DataFrame, cand_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Generate candidate pairs (S1, S2/S3) using union of multiple blocking strategies.
    Both s1_df and cand_df must already have normalized columns.
    """
    s1_id_col = 'entity_id'
    cand_id_col = 'entity_id'
    
    pairs_list = []
    
    # 1. Phonetic Blocking on Name
    print(f"[{source_name}] Phonetic blocking...")
    s1_phonetic = pd.DataFrame({
        's1_id': s1_df[s1_id_col],
        'block_key': build_phonetic_blocks(s1_df['norm_name'])
    }).dropna()
    s1_phonetic = s1_phonetic[s1_phonetic['block_key'] != '']
    
    cand_phonetic = pd.DataFrame({
        'cand_id': cand_df[cand_id_col],
        'block_key': build_phonetic_blocks(cand_df['norm_name'])
    }).dropna()
    cand_phonetic = cand_phonetic[cand_phonetic['block_key'] != '']
    
    phonetic_pairs = pd.merge(s1_phonetic, cand_phonetic, on='block_key')[['s1_id', 'cand_id']]
    pairs_list.append(phonetic_pairs)
    
    # 2. Address Postal Code Blocking
    print(f"[{source_name}] Address blocking...")
    if 'addr_postal' in s1_df.columns and 'addr_postal' in cand_df.columns:
        s1_postal = pd.DataFrame({
            's1_id': s1_df[s1_id_col],
            'block_key': s1_df['addr_postal']
        }).dropna()
        
        cand_postal = pd.DataFrame({
            'cand_id': cand_df[cand_id_col],
            'block_key': cand_df['addr_postal']
        }).dropna()
        
        postal_pairs = pd.merge(s1_postal, cand_postal, on='block_key')[['s1_id', 'cand_id']]
        pairs_list.append(postal_pairs)
        
    # 3. Token Overlap Blocking (Longest Token)
    print(f"[{source_name}] Token blocking...")
    s1_tokens = pd.DataFrame({
        's1_id': s1_df[s1_id_col],
        'block_key': build_token_blocks(s1_df['norm_name'], max_tokens=1)
    }).dropna()
    
    cand_tokens = pd.DataFrame({
        'cand_id': cand_df[cand_id_col],
        'block_key': build_token_blocks(cand_df['norm_name'], max_tokens=1)
    }).dropna()
    
    token_pairs = pd.merge(s1_tokens, cand_tokens, on='block_key')[['s1_id', 'cand_id']]
    pairs_list.append(token_pairs)

    # 4. Character TF-IDF Top-K NearestNeighbors Blocking
    print(f"[{source_name}] TF-IDF NearestNeighbors blocking...")
    knn_pairs = build_tfidf_knn_pairs(s1_df, cand_df, top_k=15)
    pairs_list.append(knn_pairs)
    
    # Union all pairs
    all_pairs = pd.concat(pairs_list, ignore_index=True)
    all_pairs.drop_duplicates(inplace=True)
    
    return all_pairs

def evaluate_blocking(pairs: pd.DataFrame, ground_truth: pd.DataFrame, s1_total: int, cand_total: int):
    """
    Evaluate recall and reduction ratio.
    """
    # Ground truth format: source1_entity_id, matched_entity_ids (comma sep)
    # Explode ground truth
    gt = ground_truth.copy()
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('').str.split(',')
    gt_exploded = gt.explode('matched_entity_ids')
    gt_exploded = gt_exploded[gt_exploded['matched_entity_ids'] != '']
    gt_exploded.columns = ['s1_id', 'cand_id']
    
    total_true_matches = len(gt_exploded)
    
    # Inner join to find retrieved true matches
    retrieved = pd.merge(pairs, gt_exploded, on=['s1_id', 'cand_id'])
    retrieved_count = len(retrieved)
    
    recall = retrieved_count / total_true_matches if total_true_matches > 0 else 0.0
    
    total_possible_pairs = s1_total * cand_total
    actual_pairs = len(pairs)
    reduction_ratio = 1.0 - (actual_pairs / total_possible_pairs) if total_possible_pairs > 0 else 0.0
    
    print(f"Blocking Evaluation:")
    print(f"  True Matches in GT: {total_true_matches}")
    print(f"  True Matches Retrieved: {retrieved_count}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Candidate Pairs Generated: {actual_pairs}")
    print(f"  Total Possible Pairs: {total_possible_pairs}")
    print(f"  Reduction Ratio: {reduction_ratio:.6f}")
    
    return recall, reduction_ratio

if __name__ == "__main__":
    print("Testing blocking...")
    s1 = pd.read_csv('dataset_sample/train/train_source1.tsv', sep='\t')
    s2 = pd.read_csv('dataset_sample/train/train_source2.tsv', sep='\t')
    s3 = pd.read_csv('dataset_sample/train/train_source3.tsv', sep='\t')
    gt = pd.read_csv('dataset_sample/train/train_ground_truth.tsv', sep='\t')
    
    s1_norm = process_dataframe(s1)
    s2_norm = process_dataframe(s2)
    s3_norm = process_dataframe(s3)
    
    print("Generating candidate pairs for S1-S2...")
    pairs_s2 = get_candidate_pairs(s1_norm, s2_norm, "S2")
    
    print("Generating candidate pairs for S1-S3...")
    pairs_s3 = get_candidate_pairs(s1_norm, s3_norm, "S3")
    
    all_pairs = pd.concat([pairs_s2, pairs_s3], ignore_index=True)
    
    evaluate_blocking(all_pairs, gt, len(s1), len(s2) + len(s3))
