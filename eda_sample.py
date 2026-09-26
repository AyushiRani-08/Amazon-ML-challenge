import pandas as pd
import numpy as np

def do_eda(path):
    print(f"=== EDA for {path} ===")
    try:
        df = pd.read_csv(path, sep='\t')
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("First 2 rows:")
        print(df.head(2).to_dict(orient='records'))
        print("Missing values:")
        print(df.isna().sum())
        print("-----------------------------\n")
    except Exception as e:
        print(f"Error reading {path}: {e}")

do_eda('dataset_sample/train/train_source1.tsv')
do_eda('dataset_sample/train/train_source2.tsv')
do_eda('dataset_sample/train/train_source3.tsv')
do_eda('dataset_sample/train/train_ground_truth.tsv')
