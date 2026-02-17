#!/usr/bin/env python3
"""
Prepare human (hg38) data for scGrapHiC inference.

This script prepares all required data in the exact format expected by scGrapHiC,
using the existing preprocessed data from human_scGrapHiC pipeline.

Data sources:
- scHi-C pseudo-bulk: /users/ssridh26/scratch/human_scGrapHiC/GSE238001/pseudobulk_official/
- RNA features: Re-created with proper normalization
- CTCF: /users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/ctcf/
- CpG: /users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/cpg/
- Bulk Hi-C: Need to extract from .hic file

Key insight from training data analysis:
- RNA (ch 0-1): Normalized to [0, 1] using library_size_normalization
- CTCF (ch 2-3): Raw accumulated scores, NOT normalized, range [0, ~1500]
- CpG (ch 4): Raw accumulated scores, NOT normalized, range [0, ~25]
"""

import numpy as np
import pandas as pd
from pathlib import Path
import gzip
from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/users/ssridh26/scratch/human_scGrapHiC")
OUTPUT_DIR = Path("/users/ssridh26/scratch/t2_human_scgraphic")

# Input paths
SCHIC_DIR = BASE_DIR / "GSE238001/pseudobulk_official"
CTCF_DIR = BASE_DIR / "annotations/preprocessed/ctcf"
CPG_DIR = BASE_DIR / "annotations/preprocessed/cpg"
GTF_FILE = BASE_DIR / "annotations/gencode.v44.annotation.gtf"
METADATA_FILE = BASE_DIR / "GSE238001/meta-python_CD_complete_hg38_filtered.csv"

RNA_FILES = {
    1: BASE_DIR / "GSE238001/scRNA-seq/GSM7657703_RNA_CD-RNA-0722-1.tsv.gz",
    2: BASE_DIR / "GSE238001/scRNA-seq/GSM7657705_RNA_CD-RNA-0722-2.tsv.gz",
    4: BASE_DIR / "GSE238001/scRNA-seq/GSM7657707_RNA_CD-RNA-0722-4.tsv.gz"
}

# Parameters matching checkpoint
RESOLUTION = 50000
LIBRARY_SIZE = 25000
CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]  # Skip chrX for now

# Cell types
CELL_TYPES = ["HSC", "MPP", "LMPP", "MEP", "B-NK"]

# hg38 chromosome sizes
CHROM_SIZES = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
    'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
    'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
    'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
    'chr21': 46709983, 'chr22': 50818468
}

# =============================================================================
# NORMALIZATION FUNCTIONS (matching scGrapHiC exactly)
# =============================================================================

def library_size_normalization(reads, library_size=LIBRARY_SIZE):
    """
    Normalize using library size normalization.
    This matches the scGrapHiC training data preprocessing.
    """
    reads = reads.astype(np.float32)
    sum_reads = np.sum(reads)
    if sum_reads > 0:
        reads = reads / sum_reads * library_size
    reads = np.log1p(reads)
    max_val = np.max(reads)
    if max_val > 0:
        reads = reads / max_val
    return reads

# =============================================================================
# GTF PARSING
# =============================================================================

def parse_gtf_genes(gtf_file):
    """Parse GTF to get gene coordinates with strand information."""
    print("  Parsing GTF for gene coordinates...")
    genes = []
    
    with open(gtf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            chrom = fields[0]
            feature = fields[2]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]
            
            # Only use gene features
            if feature != 'gene' or chrom not in CHROMOSOMES:
                continue
            
            gene_id = None
            gene_name = None
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_id'):
                    gene_id = attr.split('"')[1].split('.')[0]
                elif attr.startswith('gene_name'):
                    gene_name = attr.split('"')[1]
            
            if gene_id:
                genes.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'strand': strand
                })
    
    print(f"    Parsed {len(genes):,} genes")
    return pd.DataFrame(genes)

# =============================================================================
# RNA PROCESSING
# =============================================================================

def load_cell_annotations(metadata_file):
    """Load official cell type annotations."""
    df = pd.read_csv(metadata_file)
    cells_by_type = defaultdict(lambda: defaultdict(list))
    
    for _, row in df.iterrows():
        cell_id = row['Unnamed: 0']
        cell_type = row['cell type']
        
        parts = cell_id.split('_')
        batch_part = parts[0]
        barcode = '_'.join(parts[1:])
        batch = int(batch_part.split('-')[-1])
        
        # Normalize cell type names
        ct_normalized = cell_type.replace('=', '_').replace('-', '_')
        cells_by_type[ct_normalized][batch].append(barcode)
    
    return cells_by_type

def create_rna_features(cell_type, cells_by_batch, gene_df, output_dir):
    """
    Create RNA features for a cell type with proper library_size normalization.
    Returns (2, N) array for +/- strands, normalized to [0, 1].
    """
    print(f"\n  Processing RNA for: {cell_type}")
    total_cells = sum(len(cells) for cells in cells_by_batch.values())
    print(f"    Total cells: {total_cells}")
    
    # Map genes to bins
    gene_df = gene_df.copy()
    gene_df['start_bin'] = gene_df['start'] // RESOLUTION
    gene_df['end_bin'] = gene_df['end'] // RESOLUTION
    
    # Create gene -> bins mapping
    gene_to_bins = {}
    for _, row in gene_df.iterrows():
        gene_id = row['gene_id']
        chrom = row['chrom']
        strand = row['strand']
        bins = list(range(row['start_bin'], row['end_bin'] + 1))
        gene_to_bins[gene_id] = {'chrom': chrom, 'strand': strand, 'bins': bins}
    
    # Aggregate expression across all cells
    gene_expression = defaultdict(float)
    cells_found = 0
    
    for batch, barcodes in cells_by_batch.items():
        if batch not in RNA_FILES:
            continue
        
        rna_file = RNA_FILES[batch]
        if not rna_file.exists():
            print(f"    Warning: RNA file not found: {rna_file}")
            continue
        
        print(f"    Loading batch {batch}...")
        barcode_set = set(barcodes)
        chunk_iter = pd.read_csv(
            rna_file,
            sep='\t',
            compression='gzip',
            header=None,
            names=['chrom', 'barcode', 'gene_id'],
            chunksize=500_000,
            dtype=str
        )
        
        batch_cells_matched = set()
        for chunk in chunk_iter:
            chunk = chunk[chunk['barcode'].isin(barcode_set)]
            if chunk.empty:
                continue
            counts = chunk.groupby('gene_id').size()
            for gene_id, count in counts.items():
                gene_id_base = gene_id.split('.')[0]
                gene_expression[gene_id_base] += count
            batch_cells_matched.update(chunk['barcode'].unique())
        
        cells_found += len(batch_cells_matched)
        print(f"      Matched {len(batch_cells_matched)} cells")
    
    print(f"    Total genes detected: {len(gene_expression):,}")
    
    # Create chromosome-level features
    ct_output_dir = output_dir / cell_type
    ct_output_dir.mkdir(parents=True, exist_ok=True)
    
    for chrom in CHROMOSOMES:
        n_bins = CHROM_SIZES[chrom] // RESOLUTION + 1
        pos_track = np.zeros(n_bins, dtype=np.float32)
        neg_track = np.zeros(n_bins, dtype=np.float32)
        
        # Distribute expression to bins
        for gene_id, expr in gene_expression.items():
            if gene_id not in gene_to_bins or expr == 0:
                continue
            
            info = gene_to_bins[gene_id]
            if info['chrom'] != chrom:
                continue
            
            bins = info['bins']
            expr_per_bin = expr / len(bins)
            
            track = pos_track if info['strand'] == '+' else neg_track
            for bin_idx in bins:
                if bin_idx < len(track):
                    track[bin_idx] += expr_per_bin
        
        # Apply library_size normalization to each strand
        pos_track = library_size_normalization(pos_track)
        neg_track = library_size_normalization(neg_track)
        
        # Stack as (2, N)
        combined = np.stack([pos_track, neg_track])
        
        # Save
        output_file = ct_output_dir / f"{chrom}_{RESOLUTION}.npy"
        np.save(output_file, combined)
    
    print(f"    Saved RNA features to: {ct_output_dir}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare human data for scGrapHiC')
    parser.add_argument('--cell-type', type=str, default=None,
                        help='Process only this cell type')
    parser.add_argument('--rna-only', action='store_true',
                        help='Only process RNA features')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Prepare Human Data for scGrapHiC")
    print("=" * 60)
    
    # Parse GTF
    print("\n[1] Parsing gene annotations...")
    gene_df = parse_gtf_genes(GTF_FILE)
    
    # Load cell annotations
    print("\n[2] Loading cell type annotations...")
    cells_by_type = load_cell_annotations(METADATA_FILE)
    
    # Create output directories
    rna_output = OUTPUT_DIR / "GSE238001/scRNA-seq"
    rna_output.mkdir(parents=True, exist_ok=True)
    
    # Process each cell type
    print("\n[3] Creating RNA features with proper normalization...")
    
    cell_types_to_process = [args.cell_type] if args.cell_type else CELL_TYPES
    
    for ct in cell_types_to_process:
        ct_normalized = ct.replace('-', '_')
        if ct_normalized in cells_by_type:
            create_rna_features(ct_normalized, cells_by_type[ct_normalized], gene_df, rna_output)
        elif ct in cells_by_type:
            create_rna_features(ct, cells_by_type[ct], gene_df, rna_output)
        else:
            print(f"  Warning: Cell type {ct} not found in annotations")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nRNA features saved to: {rna_output}")
    print(f"\nExisting data to use:")
    print(f"  scHi-C: {SCHIC_DIR}")
    print(f"  CTCF: {CTCF_DIR}")
    print(f"  CpG: {CPG_DIR}")

if __name__ == "__main__":
    main()
