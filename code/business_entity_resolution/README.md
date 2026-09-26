# Business Entity Resolution Pipeline

Amazon ML Challenge 2026

## Overview
This package contains a high-precision, end-to-end Machine Learning pipeline for resolving noisy business entities across multiple data sources (Source 1 reference matched against Source 2 and Source 3 candidates). The pipeline is calibrated specifically to maximize the macro-averaged $F_{0.5}$ metric and protect singleton entities from false merges.

---

## Directory Structure
```
code/business_entity_resolution/
├── requirements.txt         # Pinned Python package dependencies
├── README.md                # Reproduction and execution guide
└── src/
    ├── normalize.py         # Unicode, legal suffix, address regex, token sorting
    ├── blocking.py          # Union candidate generation (Soundex, Postal, Token, TF-IDF k-NN)
    ├── features.py          # String, token, numeric, character n-gram TF-IDF similarities
    ├── train_model.py       # LightGBM classifier training with negative subsampling
    ├── threshold_and_predict.py # F0.5 calibration, singleton guardrails, global resolution
    ├── evaluate.py          # Macro-averaged per-entity F0.5 scoring harness
    └── make_outputs.py      # End-to-end inference driver to generate TSV submission files
```

---

## Environment & Dependencies
Python 3.10+ is recommended. Install required dependencies:

```bash
pip install -r requirements.txt
```

Pinned dependencies:
* `pandas>=2.0.0`
* `numpy>=1.24.0`
* `scipy>=1.10.0`
* `scikit-learn>=1.3.0`
* `lightgbm>=4.0.0`
* `jellyfish>=1.0.0`
* `datasketch>=2.0.0`

---

## How to Reproduce End-to-End

### Step 1: Model Training
To train the LightGBM matching model from the training dataset:
```bash
python src/train_model.py \
    --s1 ../../dataset/train/train_source1.tsv \
    --s2 ../../dataset/train/train_source2.tsv \
    --s3 ../../dataset/train/train_source3.tsv \
    --gt ../../dataset/train/train_ground_truth.tsv \
    --model ../../model/lgbm_model.pkl
```

### Step 2: Test Inference & Submission Generation
To run full inference on the unseen test dataset and generate `candidate_pairs.tsv` and `matching_results.tsv`:
```bash
python src/make_outputs.py \
    --s1 ../../dataset/test/test_source1.tsv \
    --s2 ../../dataset/test/test_source2.tsv \
    --s3 ../../dataset/test/test_source3.tsv \
    --model ../../model/lgbm_model.pkl \
    --out ../../output \
    --threshold 0.65
```

### Step 3: Validate Outputs
Verify that both output files adhere strictly to challenge formatting requirements:
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
When validation passes, `output/matching_results.tsv` is ready to upload directly to the portal!
