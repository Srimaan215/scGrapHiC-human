#!/usr/bin/env python3
"""
Create corrected NPZ files with bulk Hi-C masked in sparse regions.

This prevents the model from hallucinating contacts in regions where
ground truth is empty but K562 bulk Hi-C has signal.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


def mask_bulk_in_sparse_regions(
    input_npz,
    output_npz,
    target_threshold=0.05,
    bulk_threshold=0.1
):
    """
    Mask bulk Hi-C to zero in regions where target is sparse.
    
    Args:
        input_npz: Path to input NPZ file
        output_npz: Path to save corrected NPZ
        target_threshold: Max mean value in target to consider "sparse"
        bulk_threshold: Min mean value in bulk to consider "has signal"
    """
    print(f"Processing: {input_npz}")
    print("=" * 70)
    
    data = np.load(input_npz, allow_pickle=True)
    
    # Load all arrays
    node_features = data['node_features']
    targets = data['targets']
    pes = data['pes']
    bulk_hics = data['bulk_hics'].copy()  # Copy so we can modify
    indexes = data['indexes']
    metadatas = data['metadatas']
    
    print(f"Total samples: {len(targets)}")
    
    # Find and mask problematic samples
    masked_count = 0
    
    for i in tqdm(range(len(targets)), desc="Masking sparse regions"):
        target = targets[i, 0]
        bulk = bulk_hics[i, 0]
        
        # Calculate sparsity
        target_ut = target[np.triu_indices(target.shape[0], k=1)]
        bulk_ut = bulk[np.triu_indices(bulk.shape[0], k=1)]
        
        target_mean = target_ut.mean()
        bulk_mean = bulk_ut.mean()
        
        # Mask bulk if target is sparse but bulk has signal
        if target_mean < target_threshold and bulk_mean > bulk_threshold:
            bulk_hics[i, 0] = np.zeros_like(bulk)
            masked_count += 1
    
    print(f"\nMasked {masked_count} samples ({100*masked_count/len(targets):.1f}%)")
    
    # Save corrected dataset
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to: {output_npz}")
    np.savez(
        output_npz,
        node_features=node_features,
        targets=targets,
        pes=pes,
        bulk_hics=bulk_hics,  # Modified bulk with masked regions
        indexes=indexes,
        metadatas=metadatas
    )
    
    print("✓ Done!")
    print()


def main():
    # Process all cell types
    cell_types = ['HSC', 'MPP', 'LMPP']
    base_dir = Path('/users/ssridh26/scratch/t2_human_scgraphic/processed')
    
    for cell_type in cell_types:
        input_file = base_dir / f"{cell_type}_inference.npz"
        output_file = base_dir / f"{cell_type}_inference_corrected.npz"
        
        if input_file.exists():
            mask_bulk_in_sparse_regions(input_file, output_file)
        else:
            print(f"File not found: {input_file}\n")
    
    print("=" * 70)
    print("ALL FILES CORRECTED")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Combine corrected files:")
    print("   python combine_cell_types.py \\")
    print("       --cell_types HSC MPP LMPP \\")
    print("       --input_dir /users/ssridh26/scratch/t2_human_scgraphic/processed \\")
    print("       --output /users/ssridh26/scratch/t2_human_scgraphic/processed/combined_corrected.npz \\")
    print("       --suffix _corrected")
    print("\n2. Create splits and fine-tune with corrected data")


if __name__ == '__main__':
    main()
