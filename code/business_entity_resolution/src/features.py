import pandas as pd
import numpy as np
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

def compute_string_similarities(s1_series: pd.Series, cand_series: pd.Series) -> pd.DataFrame:
    """
    Computes Jaro-Winkler and Levenshtein ratio.
    """
    # handle NaNs
    s1 = s1_series.fillna('').astype(str).values
    s2 = cand_series.fillna('').astype(str).values
    
    # We use a vectorized approach using numpy where possible, or list comprehension which is faster than pandas apply
    # Jellyfish gives exact algorithms.
    jaro_winkler = [jellyfish.jaro_winkler_similarity(a, b) for a, b in zip(s1, s2)]
    
    # Levenshtein distance normalized to [0, 1] similarity
    # similarity = 1 - (distance / max(len(a), len(b)))
    lev_sim = []
    for a, b in zip(s1, s2):
        if not a and not b:
            lev_sim.append(1.0)
        elif not a or not b:
            lev_sim.append(0.0)
        else:
            dist = jellyfish.levenshtein_distance(a, b)
            m_len = max(len(a), len(b))
            lev_sim.append(1.0 - (dist / m_len))
            
    return pd.DataFrame({
        'jaro_winkler': jaro_winkler,
        'lev_ratio': lev_sim
    })

def compute_token_jaccard(s1_series: pd.Series, cand_series: pd.Series) -> pd.Series:
    """
    Computes token Jaccard similarity.
    """
    def jaccard(a, b):
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))
        
    return pd.Series([jaccard(a, b) for a, b in zip(s1_series.fillna(''), cand_series.fillna(''))])
    
def compute_numeric_overlap(s1_series: pd.Series, cand_series: pd.Series) -> pd.Series:
    """
    Computes numeric-token overlap in address.
    """
    import re
    def extract_numbers(s):
        return set(re.findall(r'\d+', s))
        
    def overlap(a, b):
        set_a = extract_numbers(a)
        set_b = extract_numbers(b)
        if not set_a or not set_b:
            return 0.0 # If one has no numbers, we can't be sure they overlap, so 0 (or neutral)
        return len(set_a.intersection(set_b)) / min(len(set_a), len(set_b))
        
    return pd.Series([overlap(a, b) for a, b in zip(s1_series.fillna(''), cand_series.fillna(''))])

def compute_tfidf_cosine(s1_ids: pd.Series, cand_ids: pd.Series, s1_text: pd.Series, cand_text: pd.Series) -> pd.Series:
    """
    Vectorized computation of TF-IDF cosine similarity.
    Fits TF-IDF on the union of S1 and Candidate text.
    """
    # Create an id-to-text mapping for all unique entities to build vocabulary
    all_text = pd.concat([s1_text, cand_text]).unique()
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    vectorizer.fit(all_text)
    
    # We can transform in batches or all at once
    # For large datasets, directly transforming the aligned arrays
    s1_vec = vectorizer.transform(s1_text.fillna(''))
    cand_vec = vectorizer.transform(cand_text.fillna(''))
    
    # Cosine similarity for each pair is the dot product of their respective rows
    # (since TfidfVectorizer produces L2 normalized vectors by default)
    cosine_sim = s1_vec.multiply(cand_vec).sum(axis=1).A1
    
    return pd.Series(cosine_sim)

def generate_features(pairs: pd.DataFrame, s1_df: pd.DataFrame, cand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all pairwise features.
    """
    # Merge text columns needed for features
    merged = pairs.merge(
        s1_df[['entity_id', 'norm_name', 'norm_address', 'country']], 
        left_on='s1_id', right_on='entity_id', how='left'
    ).rename(columns={'norm_name': 's1_name', 'norm_address': 's1_addr', 'country': 's1_country'}).drop('entity_id', axis=1)
    
    merged = merged.merge(
        cand_df[['entity_id', 'norm_name', 'norm_address', 'country']], 
        left_on='cand_id', right_on='entity_id', how='left'
    ).rename(columns={'norm_name': 'cand_name', 'norm_address': 'cand_addr', 'country': 'cand_country'}).drop('entity_id', axis=1)
    
    features = pd.DataFrame(index=merged.index)
    
    # Country match
    features['country_match'] = (merged['s1_country'] == merged['cand_country']).astype(int)
    
    # Name string similarities
    print("Computing name string similarities...")
    name_sims = compute_string_similarities(merged['s1_name'], merged['cand_name'])
    features['name_jaro_winkler'] = name_sims['jaro_winkler']
    features['name_lev_ratio'] = name_sims['lev_ratio']
    features['name_jaccard'] = compute_token_jaccard(merged['s1_name'], merged['cand_name'])
    
    # Address string similarities
    print("Computing address string similarities...")
    addr_sims = compute_string_similarities(merged['s1_addr'], merged['cand_addr'])
    features['addr_jaro_winkler'] = addr_sims['jaro_winkler']
    features['addr_lev_ratio'] = addr_sims['lev_ratio']
    features['addr_jaccard'] = compute_token_jaccard(merged['s1_addr'], merged['cand_addr'])
    
    # Address numeric overlap
    print("Computing numeric overlap...")
    features['addr_num_overlap'] = compute_numeric_overlap(merged['s1_addr'], merged['cand_addr'])
    
    # TF-IDF cosine
    print("Computing TF-IDF cosine for names...")
    features['name_tfidf_cosine'] = compute_tfidf_cosine(
        merged['s1_id'], merged['cand_id'], merged['s1_name'], merged['cand_name']
    )
    
    print("Computing TF-IDF cosine for addresses...")
    features['addr_tfidf_cosine'] = compute_tfidf_cosine(
        merged['s1_id'], merged['cand_id'], merged['s1_addr'], merged['cand_addr']
    )
    
    # Keep IDs for tracking
    result = pd.concat([pairs[['s1_id', 'cand_id']].reset_index(drop=True), features], axis=1)
    
    return result

if __name__ == "__main__":
    from blocking import get_candidate_pairs
    from normalize import process_dataframe
    import time
    
    print("Testing feature generation...")
    s1 = pd.read_csv('dataset_sample/train/train_source1.tsv', sep='\t').head(100)
    s2 = pd.read_csv('dataset_sample/train/train_source2.tsv', sep='\t').head(100)
    
    s1_norm = process_dataframe(s1)
    s2_norm = process_dataframe(s2)
    
    pairs = get_candidate_pairs(s1_norm, s2_norm, "S2").head(50) # Take small sample of pairs
    
    t0 = time.time()
    feats = generate_features(pairs, s1_norm, s2_norm)
    t1 = time.time()
    
    print(f"Features generated in {t1-t0:.2f} seconds.")
    print(feats.head())
