#!/usr/bin/env python3
"""
create_submission_zip.py
Packages the exact required folder structure for the final Amazon ML Challenge submission.
"""

import os
import zipfile
import sys
import argparse

def create_zip(team_name: str = "team_submission"):
    zip_filename = f"{team_name}_submission.zip"
    
    files_to_pack = [
        # (source_path, archive_path)
        ('output/matching_results.tsv', 'output/matching_results.tsv'),
        ('output/candidate_pairs.tsv', 'output/candidate_pairs.tsv'),
        ('code/business_entity_resolution/requirements.txt', 'code/business_entity_resolution/requirements.txt'),
        ('code/business_entity_resolution/README.md', 'code/business_entity_resolution/README.md'),
        ('Documentation_template.md', 'Documentation_template.md'),
    ]
    
    # Pack everything in src/
    src_dir = 'code/business_entity_resolution/src'
    for root, _, files in os.walk(src_dir):
        if '__pycache__' in root:
            continue
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, start='.')
            files_to_pack.append((full_path, rel_path.replace('\\', '/')))
            
    print(f"Creating submission package: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for src_path, arc_path in files_to_pack:
            if os.path.exists(src_path):
                zipf.write(src_path, arcname=arc_path)
                print(f"  + Added: {arc_path}")
            else:
                print(f"  ! Missing required file: {src_path}")
                
    print(f"\nSuccessfully generated {zip_filename} ({os.path.getsize(zip_filename)} bytes).")
    print("Ready for upload!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--team', default='my_team', help='Your team name')
    args = parser.parse_args()
    create_zip(args.team)
