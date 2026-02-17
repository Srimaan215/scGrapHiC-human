#!/usr/bin/env python3
"""
Check where bulk Hi-C and ground truth are mismatched.

This diagnostic script identifies regions where K562 bulk Hi-C has signal
but HSC/MPP/LMPP ground truth is empty, leading to hallucinated predictions.
"""

import numpy as np
import sys
from pathlib import Path

def check_bulk_mismatch(npz_file):
    """
    Identify samples where bulk Hi-C has signal but target is sparse.
    
    These are regions where the model will hallucinate contacts.
    """
    print(f"Analyzing: {npz_file}")
    print("=" * 70)
    
    data = np.load(npz_file, allow_pickle=True)
    
    targets = data['targets']  # Ground truth (N, 1, 128, 128)
    bulk_hics = data['bulk_hics']  # Bulk Hi-C (N, 1, 128, 128)
    indexes = data['indexes']  # (N, 4): chr, size, start, end
    
    print(f"Total samples: {len(targets)}")
    print()
    
    mismatches = []
    
    for i in range(len(targets)):
        target = targets[i, 0]  # (128, 128)
        bulk = bulk_hics[i, 0]  # (128, 128)
        
        # Get upper triangle (excluding diagonal)
        target_ut = target[np.triu_indices(target.shape[0], k=1)]
        bulk_ut = bulk[np.triu_indices(bulk.shape[0], k=1)]
        
        # Calculate sparsity
        target_mean = target_ut.mean()
        bulk_mean = bulk_ut.mean()
        
        target_nonzero_pct = 100 * (target_ut > 0).sum() / len(target_ut)
        bulk_nonzero_pct = 100 * (bulk_ut > 0).sum() / len(bulk_ut)
        
        # Flag mismatches: bulk has signal, target is sparse
        if bulk_mean > 0.1 and target_mean < 0.05:
            idx = indexes[i]
            mismatches.append({
                'sample_id': i,
                'chr': int(idx[0]),
                'start': int(idx[2]),
                'target_mean': target_mean,
                'bulk_mean': bulk_mean,
                'target_nonzero%': target_nonzero_pct,
                'bulk_nonzero%': bulk_nonzero_pct
            })
    
    print(f"Mismatched samples (bulk has signal, target sparse): {len(mismatches)}")
    print(f"Percentage: {100*len(mismatches)/len(targets):.1f}%")
    print()
    
    if mismatches:
        print("Top 20 worst mismatches:")
        print(f"{'Sample':<8} {'Chr':<5} {'Start':<8} {'Target_mean':<12} {'Bulk_mean':<12} {'Target%':<10} {'Bulk%':<10}")
        print("-" * 80)
        
        # Sort by ratio of bulk/target
        mismatches_sorted = sorted(mismatches, key=lambda x: x['bulk_mean'] / max(x['target_mean'], 0.001), reverse=True)
        
        for m in mismatches_sorted[:20]:
            print(f"{m['sample_id']:<8} {m['chr']:<5} {m['start']:<8} {m['target_mean']:<12.4f} {m['bulk_mean']:<12.4f} {m['target_nonzero%']:<10.1f} {m['bulk_nonzero%']:<10.1f}")
    
    return mismatches


if __name__ == '__main__':
    npz_files = [
        '/users/ssridh26/scratch/t2_human_scgraphic/processed/HSC_inference.npz',
        '/users/ssridh26/scratch/t2_human_scgraphic/processed/MPP_inference.npz',
        '/users/ssridh26/scratch/t2_human_scgraphic/processed/LMPP_inference.npz',
    ]
    
    all_mismatches = {}
    
    for npz_file in npz_files:
        if Path(npz_file).exists():
            cell_type = Path(npz_file).stem.split('_')[0]
            mismatches = check_bulk_mismatch(npz_file)
            all_mismatches[cell_type] = mismatches
            print()
        else:
            print(f"File not found: {npz_file}\n")
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for cell_type, mismatches in all_mismatches.items():
        print(f"{cell_type}: {len(mismatches)} mismatched samples")
