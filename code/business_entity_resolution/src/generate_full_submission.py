#!/usr/bin/env python3
"""
generate_full_submission.py
High-performance, memory-efficient pipeline for the full 1.73M Amazon ML Challenge test set.
Streams and indexes multi-million record datasets without out-of-memory errors.
"""

import os
import sys
import time
import re
import csv
import jellyfish

def clean_name(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    s = s.lower()
    # Strip non-alphanumeric
    s = re.sub(r'[^\w\s]', ' ', s)
    # Strip legal suffixes
    s = re.sub(r'\b(inc|incorporated|corp|corporation|llc|llp|pvt|ltd|limited|co|company|gmbh|sa|plc)\b', ' ', s)
    return ' '.join(s.split())

def token_sorted(s: str) -> str:
    return ' '.join(sorted(s.split()))

def extract_postal(addr: str) -> str:
    if not addr or not isinstance(addr, str):
        return ""
    m = re.search(r'\b(\d{5,6})\b', addr)
    return m.group(1) if m else ""

def build_candidate_index(test_dir: str):
    """
    Builds a high-precision inverted index from Source 2 and Source 3 files.
    """
    index = {}
    total_indexed = 0
    
    for src_name, filename in [('S2', 'test_source2.tsv'), ('S3', 'test_source3.tsv')]:
        filepath = os.path.join(test_dir, filename)
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found!")
            sys.exit(1)
            
        print(f"Reading and indexing {filename}...")
        t0 = time.time()
        chunksize = 500000
        count = 0
        
        for chunk in pd.read_csv(filepath, sep='\t', chunksize=chunksize, usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str):
            for row in chunk.itertuples(index=False):
                c_id, b_name, b_addr, country = row[0], str(row[1]), str(row[2]), str(row[3]).lower()
                c_clean = clean_name(b_name)
                if not c_clean:
                    continue
                    
                tokens = c_clean.split()
                first_token = tokens[0] if tokens else ""
                postal = extract_postal(b_addr)
                
                # Key 1: Exact cleaned name + country
                k1 = f"E:{c_clean}||{country}"
                if k1 not in index:
                    index[k1] = []
                if len(index[k1]) < 3:
                    index[k1].append((c_id, c_clean, b_addr))
                    
                # Key 2: Token-sorted cleaned name + country
                if len(tokens) > 1:
                    k2 = f"S:{token_sorted(c_clean)}||{country}"
                    if k2 not in index:
                        index[k2] = []
                    if len(index[k2]) < 3:
                        index[k2].append((c_id, c_clean, b_addr))
                        
                # Key 3: Postal code + first token + country
                if postal and len(first_token) > 3:
                    k3 = f"P:{postal}||{first_token}||{country}"
                    if k3 not in index:
                        index[k3] = []
                    if len(index[k3]) < 2:
                        index[k3].append((c_id, c_clean, b_addr))
                        
            count += len(chunk)
            print(f"  Indexed {count:,} records from {filename}...")
            
        total_indexed += count
        print(f"Finished {filename} ({count:,} records) in {time.time()-t0:.1f}s")
        
    print(f"Total candidate records indexed: {total_indexed:,}. Unique keys: {len(index):,}")
    return index

def run_pipeline(test_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    matching_out = os.path.join(output_dir, 'matching_results.tsv')
    candidate_out = os.path.join(output_dir, 'candidate_pairs.tsv')
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    
    if not os.path.exists(s1_path):
        print(f"Error: {s1_path} not found!")
        sys.exit(1)
        
    # 1. Build Index
    index = build_candidate_index(test_dir)
    
    # 2. Stream test_source1 and generate outputs line-by-line
    print(f"\nProcessing {s1_path} and writing to {matching_out} and {candidate_out}...")
    t_start = time.time()
    
    with open(s1_path, 'r', encoding='utf-8') as f_in, \
         open(matching_out, 'w', encoding='utf-8', newline='') as f_match, \
         open(candidate_out, 'w', encoding='utf-8', newline='') as f_cand:
         
        reader = csv.DictReader(f_in, delimiter='\t')
        
        # Write headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        processed_count = 0
        matched_count = 0
        candidate_count = 0
        
        # Track 1-to-1 candidate assignment to prevent duplicates across S1
        used_candidates = set()
        
        for row in reader:
            s1_id = row['entity_id']
            b_name = row.get('business_name', '')
            b_addr = row.get('business_address', '')
            country = str(row.get('country', '')).lower()
            
            s1_clean = clean_name(b_name)
            tokens = s1_clean.split()
            first_token = tokens[0] if tokens else ""
            postal = extract_postal(b_addr)
            
            # Lookup candidate keys
            keys = [f"E:{s1_clean}||{country}"]
            if len(tokens) > 1:
                keys.append(f"S:{token_sorted(s1_clean)}||{country}")
            if postal and len(first_token) > 3:
                keys.append(f"P:{postal}||{first_token}||{country}")
                
            raw_cands = []
            for k in keys:
                if k in index:
                    raw_cands.extend(index[k])
                    
            # Deduplicate candidates for this S1
            seen_cand_ids = set()
            s1_cands = []
            for c_id, c_clean, c_addr in raw_cands:
                if c_id not in seen_cand_ids:
                    seen_cand_ids.add(c_id)
                    s1_cands.append((c_id, c_clean, c_addr))
                    
            cand_id_list = [c[0] for c in s1_cands]
            
            # Match decision logic (F0.5 precision weighted)
            matched_id_list = []
            for c_id, c_clean, c_addr in s1_cands:
                if c_id in used_candidates:
                    continue
                    
                # High-precision verification
                jw_name = jellyfish.jaro_winkler_similarity(s1_clean, c_clean)
                
                # Check for match:
                # 1. Exact cleaned name OR
                # 2. Token-sorted name match OR
                # 3. High Jaro-Winkler >= 0.85
                if s1_clean == c_clean or token_sorted(s1_clean) == token_sorted(c_clean) or jw_name >= 0.85:
                    matched_id_list.append(c_id)
                    used_candidates.add(c_id)
                    
            # Write to files
            cand_str = ','.join(cand_id_list)
            match_str = ','.join(matched_id_list)
            
            f_cand.write(f"{s1_id}\t{cand_str}\n")
            f_match.write(f"{s1_id}\t{match_str}\n")
            
            processed_count += 1
            if cand_id_list:
                candidate_count += 1
            if matched_id_list:
                matched_count += 1
                
            if processed_count % 250000 == 0:
                elapsed = time.time() - t_start
                print(f"  Processed {processed_count:,} / 1,732,544 ({processed_count/1732544*100:.1f}%) in {elapsed:.1f}s | Matches: {matched_count:,}")
                
    elapsed_total = time.time() - t_start
    print(f"\nProcessing complete in {elapsed_total:.1f}s ({elapsed_total/60:.2f} mins)!")
    print(f"Total S1 entities written: {processed_count:,}")
    print(f"S1 entities with candidate pairs: {candidate_count:,}")
    print(f"S1 entities with matches predicted: {matched_count:,}")
    print(f"Singletons (empty match list): {processed_count - matched_count:,}")

if __name__ == '__main__':
    import pandas as pd
    test_dir = 'dataset/test'
    output_dir = 'output'
    run_pipeline(test_dir, output_dir)
