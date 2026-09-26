import pandas as pd
import unicodedata
import re
import numpy as np

def normalize_text(series: pd.Series) -> pd.Series:
    """
    Unicode NFKC normalization, casefold, punctuation stripping.
    """
    if series is None:
        return series
    
    # Fill NaN with empty string
    s = series.fillna('').astype(str)
    
    # Unicode NFKC normalization
    s = s.apply(lambda x: unicodedata.normalize('NFKC', x) if x else x)
    
    # Casefold
    s = s.str.casefold()
    
    # Punctuation stripping (except spaces)
    # Using regex to remove all non-alphanumeric characters except spaces
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    
    # Replace multiple spaces with single space and strip
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def normalize_legal_suffixes(series: pd.Series) -> pd.Series:
    """
    Legal-suffix normalization table.
    """
    # Dictionary for generic legal entity normalizations
    suffix_map = {
        r'\bcorp\b': 'corporation',
        r'\binc\b': 'incorporated',
        r'\bllc\b': 'limited liability company',
        r'\bllp\b': 'limited liability partnership',
        r'\bpvt\b': 'private',
        r'\bpty\b': 'proprietary',
        r'\bltd\b': 'limited',
        r'\bco\b': 'company',
        r'\bgmbh\b': 'gesellschaft mit beschrankter haftung',
        r'\bag\b': 'aktiengesellschaft',
        r'\bsa\b': 'societe anonyme',
        r'\bplc\b': 'public limited company',
        r'\band\b': '&'
    }
    
    s = series.copy()
    for pattern, repl in suffix_map.items():
        s = s.str.replace(pattern, repl, regex=True)
        
    # Standardize ampersand
    s = s.str.replace('&', ' and ')
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def normalize_address(series: pd.Series) -> pd.Series:
    """
    Address abbreviation expansion (Rd->Road, St->Street).
    """
    address_map = {
        r'\brd\b': 'road',
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bsq\b': 'square',
        r'\bste\b': 'suite',
        r'\bapt\b': 'apartment',
        r'\bfl\b': 'floor'
    }
    
    s = series.copy()
    for pattern, repl in address_map.items():
        s = s.str.replace(pattern, repl, regex=True)
    
    return s

def extract_address_components(series: pd.Series) -> pd.DataFrame:
    """
    Attempt to extract number/street/city/postal where present.
    Very basic heuristic regex since addresses are free text globally.
    """
    # Number: usually at the beginning
    # Postal: 4-6 digits or alphanumeric format
    # City/Street: hard to separate without a geocoder. 
    # We will just extract digits as numbers, and specific patterns for postal codes.
    
    df = pd.DataFrame(index=series.index)
    
    # Extract numbers (address number)
    df['addr_number'] = series.str.extract(r'(^\d+)')
    
    # Extract possible postal codes (e.g., 5-6 digits, or US zip code, UK format approx)
    # 5 digits (US/France/Germany) or 6 digits (India/China)
    df['addr_postal'] = series.str.extract(r'\b(\d{5,6})\b')
    
    # We leave street/city mostly in the normalized full address as token features will handle it.
    
    return df

def token_sort_text(series: pd.Series) -> pd.Series:
    """
    Sort tokens alphabetically to handle word-order permutations.
    e.g. 'Orelee Barbershop' <-> 'Barbershop Orelee'
    """
    if series is None:
        return series
    return series.fillna('').astype(str).apply(lambda x: ' '.join(sorted(x.split())))

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all normalizations to a DataFrame.
    """
    df = df.copy()
    
    if 'business_name' in df.columns:
        df['norm_name'] = normalize_text(df['business_name'])
        df['norm_name'] = normalize_legal_suffixes(df['norm_name'])
        df['token_sorted_name'] = token_sort_text(df['norm_name'])
        
    if 'business_address' in df.columns:
        df['norm_address'] = normalize_text(df['business_address'])
        df['norm_address'] = normalize_address(df['norm_address'])
        
        # Extract components
        components = extract_address_components(df['norm_address'])
        df['addr_number'] = components['addr_number']
        df['addr_postal'] = components['addr_postal']
        
    return df

if __name__ == "__main__":
    # Test with sample data
    print("Testing normalization...")
    df = pd.read_csv('dataset_sample/train/train_source1.tsv', sep='\t')
    norm_df = process_dataframe(df.head(10))
    print(norm_df[['business_name', 'norm_name', 'token_sorted_name', 'business_address', 'norm_address', 'addr_number', 'addr_postal']])
