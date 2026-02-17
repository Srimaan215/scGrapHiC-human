#!/usr/bin/env python3
"""
Apply augmentation strategies to combined NPZ dataset.

Implements:
1. Contact downsampling (simulate lower coverage)
2. Poisson noise injection (sequencing noise)
3. Diagonal masking (missing data simulation)

Usage:
    python augment_dataset.py \
        --input /path/to/combined_multi_celltype.npz \
        --output /path/to/augmented.npz \
        --augmentations downsample poisson \
        --downsample_rates 0.7 0.5 \
        --poisson_factor 0.1
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def downsample_contacts(matrix, downsample_rate=0.7, seed=None):
    """
    Simulate lower sequencing coverage by randomly dropping contacts.
    
    Args:
        matrix: Hi-C contact matrix (128, 128)
        downsample_rate: Fraction of contacts to keep (0.7 = keep 70%)
        seed: Random seed for reproducibility
        
    Returns:
        Downsampled matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create binary mask: 1 = keep contact, 0 = drop
    mask = np.random.binomial(1, downsample_rate, matrix.shape)
    
    # Apply mask
    downsampled = matrix * mask
    
    return downsampled


def add_poisson_noise(matrix, noise_factor=0.1, seed=None):
    """
    Add Poisson noise to contact matrix.
    
    Hi-C contact counts follow a Poisson distribution. This adds
    biologically realistic sequencing noise.
    
    Args:
        matrix: Hi-C contact matrix (128, 128), range [0, 1]
        noise_factor: Amount of noise to add (0.1 = 10% variation)
        seed: Random seed
        
    Returns:
        Noisy matrix, renormalized to [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert from [0,1] to counts (assume average of 100 reads per bin)
    scale_factor = 100
    counts = (matrix * scale_factor).astype(np.float32)
    
    noisy = np.zeros_like(counts)
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if counts[i, j] > 0:
                # Add noise to lambda parameter
                lambda_param = max(
                    counts[i, j] * (1 + np.random.uniform(-noise_factor, noise_factor)),
                    0.1
                )
                noisy[i, j] = np.random.poisson(lambda_param)
    
    # Renormalize to [0, 1]
    if noisy.max() > 0:
        noisy = noisy / noisy.max()
    
    return noisy


def mask_diagonal_bands(matrix, max_distance=10, seed=None):
    """
    Randomly mask diagonal bands to simulate missing data.
    
    Args:
        matrix: Hi-C contact matrix (128, 128)
        max_distance: Maximum diagonal offset to mask
        seed: Random seed
        
    Returns:
        Masked matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    masked = matrix.copy()
    
    # Pick random diagonal to mask
    diag_idx = np.random.randint(-max_distance, max_distance + 1)
    
    # Mask diagonal
    if diag_idx >= 0:
        # Upper diagonal
        for i in range(min(matrix.shape[0], matrix.shape[1] - diag_idx)):
            masked[i, i + diag_idx] = 0
    else:
        # Lower diagonal
        diag_idx = -diag_idx
        for i in range(min(matrix.shape[0] - diag_idx, matrix.shape[1])):
            masked[i + diag_idx, i] = 0
    
    return masked


def augment_dataset(
    input_npz,
    output_npz,
    augmentations=['downsample', 'poisson'],
    downsample_rates=[0.7, 0.5],
    poisson_factor=0.1,
    diagonal_mask_distance=10,
    seed=42
):
    """
    Create augmented version of dataset.
    
    Args:
        input_npz: Path to combined_multi_celltype.npz
        output_npz: Path to save augmented dataset
        augmentations: List of augmentation strategies to apply
        downsample_rates: List of downsampling rates for 'downsample'
        poisson_factor: Noise factor for 'poisson'
        diagonal_mask_distance: Max distance for 'diagonal_mask'
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("AUGMENTING DATASET")
    print("=" * 70)
    print(f"Input: {input_npz}")
    print(f"Output: {output_npz}")
    print(f"Augmentations: {', '.join(augmentations)}")
    print()
    
    # Load original data
    print("Loading original dataset...")
    data = np.load(input_npz, allow_pickle=True)
    
    orig_samples = len(data['targets'])
    print(f"Original samples: {orig_samples}")
    print(f"Original shape: {data['targets'].shape}")
    print()
    
    # Start with original data
    all_features = [data['node_features']]
    all_targets = [data['targets']]
    all_pes = [data['pes']]
    all_bulk = [data['bulk_hics']]
    all_idx = [data['indexes']]
    all_meta = [data['metadatas']]
    
    np.random.seed(seed)
    
    # Apply augmentations
    if 'downsample' in augmentations:
        print(f"Applying contact downsampling with rates: {downsample_rates}")
        for rate in downsample_rates:
            print(f"  Downsampling to {rate*100:.0f}% coverage...")
            downsampled_targets = []
            
            for i in tqdm(range(len(data['targets'])), desc=f"  Rate {rate}"):
                target = data['targets'][i, 0]  # (128, 128)
                downsampled = downsample_contacts(target, rate, seed=seed+i)
                downsampled_targets.append(downsampled[np.newaxis, ...])  # (1, 128, 128)
            
            # Keep features the same, update targets
            all_features.append(data['node_features'])
            all_targets.append(np.array(downsampled_targets))
            all_pes.append(data['pes'])
            all_bulk.append(data['bulk_hics'])
            all_idx.append(data['indexes'])
            all_meta.append(data['metadatas'])
            
            print(f"    Created {len(downsampled_targets)} augmented samples")
        print()
    
    if 'poisson' in augmentations:
        print(f"Applying Poisson noise (factor={poisson_factor})...")
        noisy_targets = []
        
        for i in tqdm(range(len(data['targets'])), desc="  Adding noise"):
            target = data['targets'][i, 0]  # (128, 128)
            noisy = add_poisson_noise(target, poisson_factor, seed=seed+i+10000)
            noisy_targets.append(noisy[np.newaxis, ...])  # (1, 128, 128)
        
        all_features.append(data['node_features'])
        all_targets.append(np.array(noisy_targets))
        all_pes.append(data['pes'])
        all_bulk.append(data['bulk_hics'])
        all_idx.append(data['indexes'])
        all_meta.append(data['metadatas'])
        
        print(f"  Created {len(noisy_targets)} augmented samples")
        print()
    
    if 'diagonal_mask' in augmentations:
        print(f"Applying diagonal masking (max_distance={diagonal_mask_distance})...")
        masked_targets = []
        
        for i in tqdm(range(len(data['targets'])), desc="  Masking"):
            target = data['targets'][i, 0]  # (128, 128)
            masked = mask_diagonal_bands(target, diagonal_mask_distance, seed=seed+i+20000)
            masked_targets.append(masked[np.newaxis, ...])  # (1, 128, 128)
        
        all_features.append(data['node_features'])
        all_targets.append(np.array(masked_targets))
        all_pes.append(data['pes'])
        all_bulk.append(data['bulk_hics'])
        all_idx.append(data['indexes'])
        all_meta.append(data['metadatas'])
        
        print(f"  Created {len(masked_targets)} augmented samples")
        print()
    
    # Combine all augmented versions
    print("Combining augmented datasets...")
    combined = {
        'node_features': np.concatenate(all_features, axis=0),
        'targets': np.concatenate(all_targets, axis=0),
        'pes': np.concatenate(all_pes, axis=0),
        'bulk_hics': np.concatenate(all_bulk, axis=0),
        'indexes': np.concatenate(all_idx, axis=0),
        'metadatas': np.concatenate(all_meta, axis=0)
    }
    
    # Save
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_npz}...")
    np.savez(output_npz, **combined)
    
    # Summary
    final_samples = combined['targets'].shape[0]
    augmentation_factor = final_samples / orig_samples
    
    print()
    print("=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Original samples:  {orig_samples}")
    print(f"Augmented samples: {final_samples}")
    print(f"Augmentation factor: {augmentation_factor:.1f}x")
    print(f"Final shape: {combined['targets'].shape}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Augment scGrapHiC dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input NPZ file (combined_multi_celltype.npz)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output augmented NPZ file')
    parser.add_argument('--augmentations', nargs='+',
                        choices=['downsample', 'poisson', 'diagonal_mask'],
                        default=['downsample', 'poisson'],
                        help='Which augmentations to apply')
    parser.add_argument('--downsample_rates', nargs='+', type=float,
                        default=[0.7, 0.5],
                        help='Downsampling rates (e.g., 0.7 0.5)')
    parser.add_argument('--poisson_factor', type=float, default=0.1,
                        help='Poisson noise factor (default: 0.1)')
    parser.add_argument('--diagonal_mask_distance', type=int, default=10,
                        help='Max diagonal distance for masking (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    augment_dataset(
        args.input,
        args.output,
        augmentations=args.augmentations,
        downsample_rates=args.downsample_rates,
        poisson_factor=args.poisson_factor,
        diagonal_mask_distance=args.diagonal_mask_distance,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
