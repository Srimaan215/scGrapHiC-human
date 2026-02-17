#!/usr/bin/env python3
"""
Combine multiple cell type NPZ files into a single training dataset.
This allows fine-tuning on multiple cell types simultaneously for better generalization.
"""

import numpy as np
import argparse
from pathlib import Path


def combine_npz_files(cell_types, input_dir, output_file, suffix=''):
    """
    Combine multiple cell type NPZ files into a single dataset.
    
    Args:
        cell_types: List of cell type names (e.g., ['HSC', 'MPP', 'LMPP'])
        input_dir: Directory containing the individual NPZ files
        output_file: Output path for combined NPZ
        suffix: Optional suffix for input files (e.g., '_corrected')
    """
    print("="*70)
    print("COMBINING CELL TYPE NPZ FILES")
    print("="*70)
    print(f"Cell types: {', '.join(cell_types)}")
    print(f"Output: {output_file}")
    if suffix:
        print(f"Using suffix: {suffix}")
    
    all_node_features = []
    all_targets = []
    all_pes = []
    all_bulk_hics = []
    all_indexes = []
    all_metadatas = []
    
    for cell_type in cell_types:
        npz_path = Path(input_dir) / f"{cell_type}_inference{suffix}.npz"
        
        if not npz_path.exists():
            print(f"\n⚠️  Warning: {npz_path} not found, skipping...")
            continue
        
        print(f"\nLoading {cell_type}...")
        data = np.load(npz_path, allow_pickle=True)
        
        n_samples = data['node_features'].shape[0]
        print(f"  Samples: {n_samples}")
        
        all_node_features.append(data['node_features'])
        all_targets.append(data['targets'])
        all_pes.append(data['pes'])
        all_bulk_hics.append(data['bulk_hics'])
        all_indexes.append(data['indexes'])
        all_metadatas.append(data['metadatas'])
    
    if not all_node_features:
        print("\n❌ No valid NPZ files found!")
        return
    
    # Concatenate all data
    print(f"\nCombining {len(all_node_features)} cell types...")
    combined = {
        'node_features': np.concatenate(all_node_features, axis=0),
        'targets': np.concatenate(all_targets, axis=0),
        'pes': np.concatenate(all_pes, axis=0),
        'bulk_hics': np.concatenate(all_bulk_hics, axis=0),
        'indexes': np.concatenate(all_indexes, axis=0),
        'metadatas': np.concatenate(all_metadatas, axis=0)
    }
    
    print(f"\nCombined dataset:")
    print(f"  Total samples: {combined['node_features'].shape[0]}")
    print(f"  node_features: {combined['node_features'].shape}")
    print(f"  targets: {combined['targets'].shape}")
    
    # Save combined dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path, **combined)
    print(f"\n✅ Saved combined dataset to: {output_path}")
    
    # Statistics
    print(f"\nDataset statistics:")
    unique_cells = np.unique(combined['metadatas'][:, 2])
    print(f"  Cell types present: {len(unique_cells)}")
    
    chromosomes = combined['indexes'][:, 0]
    unique_chrs, counts = np.unique(chromosomes, return_counts=True)
    print(f"  Chromosomes: {len(unique_chrs)}")
    print(f"  Samples per chromosome (mean): {counts.mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description='Combine cell type NPZ files')
    parser.add_argument('--cell_types', nargs='+', 
                        default=['HSC', 'MPP', 'LMPP'],
                        help='Cell types to combine')
    parser.add_argument('--input_dir', type=str,
                        default='/users/ssridh26/scratch/t2_human_scgraphic/processed',
                        help='Directory containing NPZ files')
    parser.add_argument('--output', type=str,
                        default='/users/ssridh26/scratch/t2_human_scgraphic/processed/combined_multi_celltype.npz',
                        help='Output file path')
    parser.add_argument('--suffix', type=str, default='',
                        help='Optional suffix for input files (e.g., "_corrected")')
    
    args = parser.parse_args()
    
    combine_npz_files(args.cell_types, args.input_dir, args.output, args.suffix)


if __name__ == '__main__':
    main()
