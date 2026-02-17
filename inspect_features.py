#!/usr/bin/env python3
"""Inspect features in NPZ files to diagnose inference issues."""

import numpy as np
import sys

def inspect_npz(filepath):
    """Load and inspect NPZ file features."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {filepath}")
    print('='*80)
    
    data = np.load(filepath, allow_pickle=True)
    
    # List all keys
    print(f"\nKeys in NPZ: {list(data.keys())}")
    
    # Inspect node_features
    if 'node_features' in data:
        features = data['node_features']
        print(f"\nNode Features Shape: {features.shape}")
        print(f"Node Features dtype: {features.dtype}")
        
        # Check for each feature dimension
        n_samples = min(5, features.shape[0])
        n_bins = features.shape[1]  # Should be 128
        n_features = features.shape[2]  # Should be 5
        
        print(f"\nFeature dimensions: samples={features.shape[0]}, bins={n_bins}, features={n_features}")
        
        # Statistics for each feature across all samples
        print(f"\nFeature Statistics (across all samples):")
        for i in range(n_features):
            feat = features[:, :, i]
            print(f"  Feature {i}: min={feat.min():.6f}, max={feat.max():.6f}, "
                  f"mean={feat.mean():.6f}, std={feat.std():.6f}, "
                  f"zeros={np.sum(feat == 0)}/{feat.size} ({100*np.sum(feat == 0)/feat.size:.1f}%)")
        
        # Show first sample details
        print(f"\nFirst Sample Details:")
        for i in range(n_features):
            feat = features[0, :, i]
            print(f"  Feature {i}: min={feat.min():.6f}, max={feat.max():.6f}, "
                  f"mean={feat.mean():.6f}, nonzero_bins={np.sum(feat > 0)}/{n_bins}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(features)):
            print(f"\n⚠️  WARNING: Found {np.sum(np.isnan(features))} NaN values!")
        if np.any(np.isinf(features)):
            print(f"\n⚠️  WARNING: Found {np.sum(np.isinf(features))} Inf values!")
        
        # Check if all features are zeros (bad)
        all_zeros = np.all(features == 0, axis=(1, 2))
        if np.any(all_zeros):
            print(f"\n⚠️  WARNING: {np.sum(all_zeros)} samples have ALL ZERO features!")
    
    # Inspect targets
    if 'targets' in data:
        targets = data['targets']
        print(f"\nTargets Shape: {targets.shape}")
        print(f"Targets dtype: {targets.dtype}")
        print(f"Targets: min={targets.min():.6f}, max={targets.max():.6f}, "
              f"mean={targets.mean():.6f}, nonzero={np.sum(targets > 0)}/{targets.size} ({100*np.sum(targets > 0)/targets.size:.2f}%)")
    
    # Inspect bulk_hics
    if 'bulk_hics' in data:
        bulk = data['bulk_hics']
        print(f"\nBulk Hi-C Shape: {bulk.shape}")
        print(f"Bulk Hi-C dtype: {bulk.dtype}")
        print(f"Bulk Hi-C: min={bulk.min():.6f}, max={bulk.max():.6f}, "
              f"mean={bulk.mean():.6f}, nonzero={np.sum(bulk > 0)}/{bulk.size} ({100*np.sum(bulk > 0)/bulk.size:.2f}%)")
    
    # Inspect position encodings
    if 'pes' in data:
        pes = data['pes']
        print(f"\nPosition Encodings Shape: {pes.shape}")
        print(f"Position Encodings: min={pes.min():.6f}, max={pes.max():.6f}, mean={pes.mean():.6f}")
    
    # Inspect metadata
    if 'metadatas' in data:
        metadatas = data['metadatas']
        print(f"\nMetadata Shape: {metadatas.shape}")
        if metadatas.size > 0:
            print(f"First 3 metadatas:")
            for i in range(min(3, len(metadatas))):
                print(f"  {i}: {metadatas[i]}")
    
    data.close()

if __name__ == "__main__":
    # Check training data
    print("\n" + "="*80)
    print("TRAINING DATA")
    print("="*80)
    inspect_npz("/users/ssridh26/scratch/t2_human_scgraphic/processed/combined_gm12878.npz")
    
    # Check inference data
    print("\n\n" + "="*80)
    print("INFERENCE DATA (GM12878)")
    print("="*80)
    
    for cell_type in ['HSC', 'MPP', 'LMPP']:
        inspect_npz(f"/users/ssridh26/scratch/t2_human_scgraphic/processed/{cell_type}_inference_gm12878.npz")
