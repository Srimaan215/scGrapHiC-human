#!/usr/bin/env python3
"""
Check sparsity of each cell type in pseudobulk_official directory.
"""
import numpy as np
import os
from pathlib import Path

base_dir = Path("/users/ssridh26/scratch/human_scGrapHiC/GSE238001/pseudobulk_official")
cell_types = [d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d)]

results = []

for ct in cell_types:
    ct_dir = base_dir / ct
    npy_files = list(ct_dir.glob("chr*.npy"))
    
    if not npy_files:
        print(f"{ct}: No chromosome files found")
        continue
    
    total_elements = 0
    zero_elements = 0
    total_contacts = 0
    
    for npy_file in npy_files:
        try:
            matrix = np.load(npy_file)
            total_elements += matrix.size
            zero_elements += (matrix == 0).sum()
            total_contacts += matrix.sum()
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            
    sparsity = (zero_elements / total_elements) * 100.0
    avg_contacts_per_chr = total_contacts / len(npy_files)
    
    results.append({
        'cell_type': ct,
        'sparsity': sparsity,
        'total_contacts': total_contacts,
        'avg_contacts_per_chr': avg_contacts_per_chr,
        'n_chromosomes': len(npy_files)
    })
    
    print(f"{ct}: {sparsity:.2f}% sparse, {total_contacts:.0f} total contacts")

# Sort by sparsity (ascending = lowest sparsity first)
results.sort(key=lambda x: x['sparsity'])

print("\n" + "="*60)
print("CELL TYPES RANKED BY SPARSITY (Lowest = Most Dense)")
print("="*60)
for i, res in enumerate(results, 1):
    print(f"{i}. {res['cell_type']:15s} - {res['sparsity']:6.2f}% sparse, {res['total_contacts']:.0f} contacts")

if len(results) >= 2:
    print(f"\n{'='*60}")
    print(f"2 LOWEST SPARSITY (MOST DENSE) CELL TYPES:")
    print(f"{'='*60}")
    print(f"1. {results[0]['cell_type']}")
    print(f"2. {results[1]['cell_type']}")
