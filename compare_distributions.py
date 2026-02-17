#!/usr/bin/env python3
"""Compare data distributions between train/val and inference sets."""

import numpy as np
from collections import Counter

def analyze_file(filepath, name):
    """Analyze chromosome and sample distribution."""
    print(f"\n{'='*80}")
    print(f"{name}: {filepath}")
    print('='*80)
    
    data = np.load(filepath, allow_pickle=True)
    metadatas = data['metadatas']
    
    # Extract chromosome info (metadata format: [tissue_id, donor_id, cell_type_id, chr_id])
    chr_ids = metadatas[:, 3]
    cell_type_ids = metadatas[:, 2]
    
    print(f"\nTotal samples: {len(metadatas)}")
    print(f"\nChromosome distribution:")
    chr_counts = Counter(chr_ids)
    for chr_id in sorted(chr_counts.keys()):
        pct = 100 * chr_counts[chr_id] / len(metadatas)
        print(f"  chr{chr_id}: {chr_counts[chr_id]:4d} ({pct:5.2f}%)")
    
    print(f"\nCell type distribution:")
    cell_counts = Counter(cell_type_ids)
    cell_type_names = {28: 'HSC', 29: 'MPP', 30: 'LMPP', 31: 'MEP', 32: 'MLP', 33: 'B_NK'}
    for cell_id in sorted(cell_counts.keys()):
        cell_name = cell_type_names.get(cell_id, f'Unknown({cell_id})')
        pct = 100 * cell_counts[cell_id] / len(metadatas)
        print(f"  {cell_name}: {cell_counts[cell_id]:4d} ({pct:5.2f}%)")
    
    data.close()
    return chr_counts, cell_counts

# Training data
SPLITS_DIR = "/users/ssridh26/scratch/t2_human_scgraphic/processed/splits_gm12878/combined"
train_chr, train_cell = analyze_file(
    f"{SPLITS_DIR}/combined_train.npz",
    "TRAINING SET"
)

# Validation data
val_chr, val_cell = analyze_file(
    f"{SPLITS_DIR}/combined_val.npz",
    "VALIDATION SET"
)

# Test data
test_chr, test_cell = analyze_file(
    f"{SPLITS_DIR}/combined_test.npz",
    "TEST SET"
)

# Inference data
hsc_chr, hsc_cell = analyze_file(
    "/users/ssridh26/scratch/t2_human_scgraphic/processed/HSC_inference_gm12878.npz",
    "INFERENCE SET (HSC)"
)

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print("\nChromosomes present:")
all_chrs = set(train_chr.keys()) | set(val_chr.keys()) | set(test_chr.keys()) | set(hsc_chr.keys())
print(f"  Train: {sorted(train_chr.keys())}")
print(f"  Val:   {sorted(val_chr.keys())}")
print(f"  Test:  {sorted(test_chr.keys())}")
print(f"  HSC Inf: {sorted(hsc_chr.keys())}")

print("\nDifferences:")
val_only = set(val_chr.keys()) - set(hsc_chr.keys())
hsc_only = set(hsc_chr.keys()) - set(val_chr.keys())
if val_only:
    print(f"  Chromosomes in Val but NOT in HSC Inference: {sorted(val_only)}")
if hsc_only:
    print(f"  Chromosomes in HSC Inference but NOT in Val: {sorted(hsc_only)}")
