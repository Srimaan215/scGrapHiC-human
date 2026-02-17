#!/usr/bin/env python3
"""
Create train/test/val split for human scGrapHiC fine-tuning.

Split strategy:
- 60% training, 20% testing, 20% validation
- Split by chromosome to avoid data leakage
- Apply Option A filtering: keep only samples with <10% negative values

Chromosome assignment:
- Training (14 chromosomes): chr1, chr2, chr3, chr4, chr5, chr6, chr7, chr8, chr10, chr11, chr14, chr15, chr18, chr20
- Validation (4 chromosomes): chr9, chr12, chr17, chr21
- Testing (4 chromosomes): chr13, chr16, chr19, chr22

This gives approximately 60:20:20 split based on chromosome sizes.
"""

import numpy as np
import argparse
from pathlib import Path


# Chromosome split assignments
TRAIN_CHROMOSOMES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 18, 20]  # 14 chromosomes (~60%)
VAL_CHROMOSOMES = [9, 12, 17, 21]  # 4 chromosomes (~20%)
TEST_CHROMOSOMES = [13, 16, 19, 22]  # 4 chromosomes (~20%)


def filter_by_negative_threshold(targets, threshold=10.0):
    """
    Filter samples to keep only those with less than threshold% negative values.
    
    Args:
        targets: Array of shape (N, 1, 128, 128)
        threshold: Maximum percentage of negative values allowed
        
    Returns:
        Boolean mask of valid samples
    """
    valid_mask = np.zeros(len(targets), dtype=bool)
    
    for i in range(len(targets)):
        t = targets[i, 0]  # (128, 128)
        t_ut = t[np.triu_indices(t.shape[0], k=1)]
        neg_pct = 100 * (t_ut < 0).sum() / len(t_ut)
        zero_pct = 100 * (t_ut == 0).sum() / len(t_ut)
        
        # Option A: <10% negative values and <1% zeros
        if neg_pct < threshold and zero_pct < 1.0:
            valid_mask[i] = True
    
    return valid_mask


def create_splits(input_npz_path, output_dir, negative_threshold=10.0, filter_training_only=True):
    """
    Create train/test/val splits from the input NPZ file.
    
    Args:
        input_npz_path: Path to the input NPZ file (e.g., HSC_inference.npz)
        output_dir: Directory to save the split NPZ files
        negative_threshold: Maximum % negative values for filtering
        filter_training_only: If True, only filter training set; keep all samples for test/val
    """
    print(f"Loading data from {input_npz_path}...")
    data = np.load(input_npz_path, allow_pickle=True)
    
    # Extract arrays
    node_features = data['node_features']
    targets = data['targets']
    pes = data['pes']
    bulk_hics = data['bulk_hics']
    indexes = data['indexes']
    metadatas = data['metadatas']
    
    print(f"Total samples: {len(targets)}")
    print(f"Node features shape: {node_features.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Get chromosome numbers from indexes (column 0)
    # Previous code: chromosomes = metadatas[:, 3]
    chromosomes = indexes[:, 0]
    
    # Create split masks based on chromosomes
    train_chr_mask = np.isin(chromosomes, TRAIN_CHROMOSOMES)
    val_chr_mask = np.isin(chromosomes, VAL_CHROMOSOMES)
    test_chr_mask = np.isin(chromosomes, TEST_CHROMOSOMES)
    
    print(f"\nChromosome-based split:")
    print(f"  Training chromosomes {TRAIN_CHROMOSOMES}: {train_chr_mask.sum()} samples")
    print(f"  Validation chromosomes {VAL_CHROMOSOMES}: {val_chr_mask.sum()} samples")
    print(f"  Test chromosomes {TEST_CHROMOSOMES}: {test_chr_mask.sum()} samples")
    
    # Apply Option A filtering
    print(f"\nApplying Option A filter (negative < {negative_threshold}%, zeros < 1%)...")
    quality_mask = filter_by_negative_threshold(targets, negative_threshold)
    print(f"  Samples passing quality filter: {quality_mask.sum()} / {len(targets)} ({100*quality_mask.sum()/len(targets):.1f}%)")
    
    # Combine masks
    if filter_training_only:
        # Only filter training data, keep all samples for test/val
        train_mask = train_chr_mask & quality_mask
        val_mask = val_chr_mask  # No quality filter
        test_mask = test_chr_mask  # No quality filter
        print("\n  Note: Quality filter applied to training set only")
    else:
        # Filter all splits
        train_mask = train_chr_mask & quality_mask
        val_mask = val_chr_mask & quality_mask
        test_mask = test_chr_mask & quality_mask
        print("\n  Note: Quality filter applied to all splits")
    
    print(f"\nFinal split sizes:")
    print(f"  Training: {train_mask.sum()} samples")
    print(f"  Validation: {val_mask.sum()} samples")
    print(f"  Test: {test_mask.sum()} samples")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits = [
        ('train', train_mask),
        ('val', val_mask),
        ('test', test_mask)
    ]
    
    # Determine output prefix from input filename if not HSC
    input_name = Path(input_npz_path).stem
    if 'combined' in input_name or 'multi' in input_name:
        file_prefix = 'combined'
    else:
        file_prefix = input_name.replace('_inference', '')
    
    for split_name, mask in splits:
        output_file = output_path / f'{file_prefix}_{split_name}.npz'
        
        # Extract data for this split
        split_data = {
            'node_features': node_features[mask],
            'targets': targets[mask],
            'pes': pes[mask],
            'bulk_hics': bulk_hics[mask],
            'indexes': indexes[mask],
            'metadatas': metadatas[mask]
        }
        
        np.savez(output_file, **split_data)
        print(f"\nSaved {split_name} split to {output_file}")
        print(f"  Samples: {mask.sum()}")
        
        # Print chromosome distribution
        split_chrs = metadatas[mask, 3]
        unique_chrs, counts = np.unique(split_chrs, return_counts=True)
        print(f"  Chromosomes: {dict(zip(unique_chrs.astype(int), counts))}")
        
        # Print quality stats
        split_targets = targets[mask]
        neg_pcts = []
        for i in range(len(split_targets)):
            t = split_targets[i, 0]
            t_ut = t[np.triu_indices(t.shape[0], k=1)]
            neg_pcts.append(100 * (t_ut < 0).sum() / len(t_ut))
        print(f"  Mean negative %: {np.mean(neg_pcts):.2f}%")
    
    print("\n" + "="*60)
    print("Split creation complete!")
    print("="*60)
    
    return train_mask, val_mask, test_mask


def main():
    parser = argparse.ArgumentParser(description='Create train/test/val splits for human scGrapHiC')
    parser.add_argument('--input', type=str, 
                        default='/users/ssridh26/scratch/t2_human_scgraphic/processed/HSC_inference.npz',
                        help='Path to input NPZ file')
    parser.add_argument('--output', type=str,
                        default='/users/ssridh26/scratch/t2_human_scgraphic/processed/splits',
                        help='Output directory for split files')
    parser.add_argument('--negative_threshold', type=float, default=10.0,
                        help='Maximum percentage of negative values allowed (default: 10)')
    parser.add_argument('--filter_all', action='store_true',
                        help='Apply quality filter to all splits, not just training')
    
    args = parser.parse_args()
    
    create_splits(
        args.input,
        args.output,
        negative_threshold=args.negative_threshold,
        filter_training_only=not args.filter_all
    )


if __name__ == '__main__':
    main()
