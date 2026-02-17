#!/usr/bin/env python3
"""
Create pseudobulk scHi-C matrices from raw pairs files.

This script:
1. Reads scHi-C pairs files for each cell type
2. Aggregates contacts into chromosome-level contact matrices
3. Saves as .npy files in the format expected by scGrapHiC

Input: GSE238001 pairs files (*.pairs.gz)
Output: Per-chromosome .npy contact matrices for each cell type
"""

import numpy as np
import pandas as pd
import gzip
import os
import re
from pathlib import Path
from collections import defaultdict
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

RESOLUTION = 50000

# hg38 chromosome sizes
CHROM_SIZES = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
    'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
    'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
    'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
    'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895
}

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]  # chr1-22

# Cell type mapping from GSE238001 sample names
# Based on the GEO metadata, files are named like:
# GSM*_HiC_CD-HiC-0722-1.pairs.gz, etc.
# Cell types are determined by the metadata


def get_cell_type_from_metadata(metadata_file, sample_id):
    """Get cell type from metadata CSV."""
    if not os.path.exists(metadata_file):
        return None
    df = pd.read_csv(metadata_file)
    # Look up sample
    match = df[df['sample_id'] == sample_id]
    if len(match) > 0:
        return match.iloc[0]['cell_type']
    return None


def parse_pairs_file(pairs_file, resolution=50000):
    """
    Parse a .pairs.gz file and return contact counts per chromosome.
    
    GAGE-seq pairs format (no header):
    chr1    pos1    chr2    pos2    cell_barcode
    e.g.: hg38_chr1   12345   hg38_chr1   67890   C6,H12
    
    Returns:
        dict: {chrom: sparse_contacts} where sparse_contacts is a dict of {(bin1, bin2): count}
    """
    contacts = defaultdict(lambda: defaultdict(int))
    
    opener = gzip.open if str(pairs_file).endswith('.gz') else open
    mode = 'rt' if str(pairs_file).endswith('.gz') else 'r'
    
    with opener(pairs_file, mode) as f:
        for line in f:
            # Skip header lines
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            
            try:
                # GAGE-seq pairs format: chr1 pos1 chr2 pos2 [barcode]
                # Chromosomes have hg38_ prefix, e.g., hg38_chr1
                chrom1_raw = parts[0]
                pos1 = int(parts[1])
                chrom2_raw = parts[2]
                pos2 = int(parts[3])
                
                # Remove hg38_ prefix if present
                chrom1 = chrom1_raw.replace('hg38_', '')
                chrom2 = chrom2_raw.replace('hg38_', '')
                
                # Only keep cis contacts (same chromosome)
                if chrom1 != chrom2:
                    continue
                
                # Only keep autosomes
                if chrom1 not in CHROMOSOMES:
                    continue
                
                # Bin positions
                bin1 = pos1 // resolution
                bin2 = pos2 // resolution
                
                # Make sure bin1 <= bin2 for symmetry
                if bin1 > bin2:
                    bin1, bin2 = bin2, bin1
                
                contacts[chrom1][(bin1, bin2)] += 1
                
            except (ValueError, IndexError):
                continue
    
    return contacts


def sparse_to_dense(sparse_contacts, chrom_size, resolution=50000):
    """Convert sparse contact dict to dense matrix."""
    n_bins = chrom_size // resolution + 1
    matrix = np.zeros((n_bins, n_bins), dtype=np.float32)
    
    for (bin1, bin2), count in sparse_contacts.items():
        if bin1 < n_bins and bin2 < n_bins:
            matrix[bin1, bin2] = count
            matrix[bin2, bin1] = count  # Symmetric
    
    return matrix


def aggregate_cells(pairs_files, cell_types, target_cell_type, resolution=50000):
    """
    Aggregate contacts from all cells of a given cell type.
    
    Args:
        pairs_files: List of paths to pairs files
        cell_types: Dict mapping pairs file -> cell type
        target_cell_type: Cell type to aggregate
        resolution: Bin size
    
    Returns:
        dict: {chrom: contact_matrix}
    """
    # Initialize contact matrices
    aggregated = {}
    for chrom in CHROMOSOMES:
        n_bins = CHROM_SIZES[chrom] // resolution + 1
        aggregated[chrom] = np.zeros((n_bins, n_bins), dtype=np.float32)
    
    # Find cells of target type
    matching_files = [f for f in pairs_files if cell_types.get(os.path.basename(f)) == target_cell_type]
    
    if not matching_files:
        print(f"  No files found for cell type: {target_cell_type}")
        return aggregated
    
    print(f"  Aggregating {len(matching_files)} cells for {target_cell_type}")
    
    for i, pf in enumerate(matching_files):
        if (i + 1) % 10 == 0:
            print(f"    Processing {i+1}/{len(matching_files)}: {os.path.basename(pf)}")
        
        contacts = parse_pairs_file(pf, resolution)
        
        for chrom, sparse in contacts.items():
            for (bin1, bin2), count in sparse.items():
                if bin1 < aggregated[chrom].shape[0] and bin2 < aggregated[chrom].shape[1]:
                    aggregated[chrom][bin1, bin2] += count
                    if bin1 != bin2:
                        aggregated[chrom][bin2, bin1] += count
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Create pseudobulk scHi-C matrices')
    parser.add_argument('--input-dir', type=str,
                        default='/users/ssridh26/scratch/human_scGrapHiC/GSE238001/scHi-C',
                        help='Directory containing pairs files')
    parser.add_argument('--output-dir', type=str,
                        default='/users/ssridh26/scratch/human_scGrapHiC/GSE238001/pseudobulk_official',
                        help='Output directory for pseudobulk matrices')
    parser.add_argument('--metadata', type=str,
                        default='/users/ssridh26/scratch/human_scGrapHiC/GSE238001/metadata.csv',
                        help='Metadata CSV with cell type annotations')
    parser.add_argument('--cell-type', type=str, default=None,
                        help='Process only this cell type (default: all)')
    parser.add_argument('--resolution', type=int, default=50000,
                        help='Bin resolution in bp')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Creating Pseudobulk scHi-C Matrices")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Resolution: {args.resolution} bp")
    print()
    
    # Find all pairs files
    pairs_files = list(Path(args.input_dir).glob("*.pairs.gz"))
    print(f"Found {len(pairs_files)} pairs files")
    
    if len(pairs_files) == 0:
        print("ERROR: No pairs files found!")
        return
    
    # Load or create cell type mapping
    # For GSE238001, we need to parse cell types from metadata or filenames
    # The files are typically named like: GSM*_HiC_CD-HiC-XXXX-X.pairs.gz
    # Cell type is usually in a separate metadata file
    
    # If no metadata, try to infer from existing directory structure or create default
    cell_types = {}
    
    if os.path.exists(args.metadata):
        print(f"Loading metadata from {args.metadata}")
        meta_df = pd.read_csv(args.metadata)
        # Assume columns: filename, cell_type
        for _, row in meta_df.iterrows():
            cell_types[row['filename']] = row['cell_type']
    else:
        print("No metadata file found. Using 'HSC' as default cell type for all cells.")
        print("To properly separate cell types, create a metadata.csv with columns: filename, cell_type")
        for pf in pairs_files:
            cell_types[os.path.basename(pf)] = 'HSC'
    
    # Get unique cell types
    unique_types = set(cell_types.values())
    print(f"Cell types: {unique_types}")
    
    # Filter to requested cell type if specified
    if args.cell_type:
        if args.cell_type in unique_types:
            unique_types = {args.cell_type}
        else:
            print(f"ERROR: Cell type '{args.cell_type}' not found in metadata")
            return
    
    # Process each cell type
    for ct in unique_types:
        print(f"\nProcessing cell type: {ct}")
        
        # Create output directory
        ct_output_dir = os.path.join(args.output_dir, ct)
        os.makedirs(ct_output_dir, exist_ok=True)
        
        # Aggregate contacts
        matrices = aggregate_cells(
            [str(pf) for pf in pairs_files],
            cell_types,
            ct,
            args.resolution
        )
        
        # Save matrices
        for chrom, matrix in matrices.items():
            output_path = os.path.join(ct_output_dir, f"{chrom}.npy")
            np.save(output_path, matrix)
            total_contacts = matrix.sum() / 2  # Divide by 2 because symmetric
            print(f"    {chrom}: {matrix.shape}, contacts={total_contacts:.0f}")
        
        print(f"  Saved to {ct_output_dir}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
