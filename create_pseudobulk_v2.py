#!/usr/bin/env python3
"""
Create pseudobulk scHi-C and scRNA-seq data for Human GSE238001.

This script processes raw scHi-C pairs and scRNA-seq transcript lists,
aggregating them into pseudobulk contact matrices (.npy) and gene expression tracks (.npy)
suitable for scGrapHiC inference.
"""

import os
import sys
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
from pathlib import Path

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

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]  # chr1-22 (Automosomes only for now as per previous scripts)

# Paths
DEFAULT_HIC_DIR = "/users/ssridh26/scratch/human_scGrapHiC_data/GSE238001/raw/scHi-C"
DEFAULT_RNA_DIR = "/users/ssridh26/scratch/human_scGrapHiC_data/GSE238001/raw/scRNA/scRNA-seq"
DEFAULT_OUTPUT_DIR = "/users/ssridh26/scratch/human_scGrapHiC/GSE238001/pseudobulk_generated"
DEFAULT_GENES_BED = "/users/ssridh26/projects/human_scGrapHiC/everything_prior_to_pairs/preprocessed_defunct/gencode.v36.genes.bed"


def load_gene_coordinates(bed_file):
    """
    Load gene coordinates from BED file.
    Returns dict: {gene_id: (chrom, bin_idx)}
    """
    print(f"Loading gene coordinates from {bed_file}...")
    gene_map = {}
    
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            
            chrom = parts[0]
            if chrom not in CHROMOSOMES: continue
            
            start = int(parts[1])
            end = int(parts[2])
            gene_id_full = parts[3]
            gene_id = gene_id_full.split('.')[0] # Remove version
            
            # Use center of gene for bin assignment
            center = (start + end) // 2
            bin_idx = center // RESOLUTION
            
            gene_map[gene_id] = (chrom, bin_idx)
            
    print(f"Loaded {len(gene_map)} genes.")
    return gene_map


def process_schic_file(pairs_file, output_dir_base):
    """Process a single scHi-C pairs file."""
    filename = os.path.basename(pairs_file)
    # Extract sample name (remove extension and prefixes if standard)
    # Example: GSM7657692_contact_K3-0208-PL2-HiC.pairs.gz -> K3-0208-PL2
    # Or keep full name for safety.
    # Let's try to extract the middle part which looks like the sample ID
    
    # Simple heuristic: Split by '_' and take parts, or regex
    # GSM7657692_contact_K3-0208-PL2-HiC.pairs.gz
    # GSM7657700_contact_mBC-HiC-0716-2.pairs.gz
    
    if "_contact_" in filename and "-HiC" in filename:
        parts = filename.split("_contact_")[1].split("-HiC")[0]
        # For mBC ones: mBC-HiC-0716-1 -> mBC (overlap?)
        # Actually, let's just use the filename without extension as the ID to be unique and safe.
        sample_id = filename.replace(".pairs.gz", "")
    else:
        sample_id = filename.replace(".pairs.gz", "")
        
    print(f"Processing scHi-C: {filename} -> ID: {sample_id}")
    
    out_dir = os.path.join(output_dir_base, "scHi-C", sample_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize matrices
    contacts = {}
    for chrom in CHROMOSOMES:
        n_bins = CHROM_SIZES[chrom] // RESOLUTION + 1
        contacts[chrom] = np.zeros((n_bins, n_bins), dtype=np.float32)
        
    # Read pairs
    opener = gzip.open if str(pairs_file).endswith('.gz') else open
    mode = 'rt' if str(pairs_file).endswith('.gz') else 'r'
    
    count = 0
    with opener(pairs_file, mode) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            
            c1 = parts[0].replace('hg38_', '')
            p1 = int(parts[1])
            c2 = parts[2].replace('hg38_', '')
            p2 = int(parts[3])
            
            if c1 != c2: continue # Cis only
            if c1 not in CHROMOSOMES: continue
            
            b1 = p1 // RESOLUTION
            b2 = p2 // RESOLUTION
            
            if b1 > b2: b1, b2 = b2, b1
            
            n_bins = contacts[c1].shape[0]
            if b1 < n_bins and b2 < n_bins:
                contacts[c1][b1, b2] += 1
                if b1 != b2:
                    contacts[c1][b2, b1] += 1
            
            count += 1
            if count % 1000000 == 0:
                print(f"  Processed {count/1000000:.1f}M pairs...", end='\r')
                
    print(f"  Total pairs processed: {count}")
    
    # Save
    for chrom, matrix in contacts.items():
        np.save(os.path.join(out_dir, f"{chrom}.npy"), matrix)
    print(f"  Saved to {out_dir}")


def process_rna_file(tsv_file, gene_map, output_dir_base):
    """Process a single scRNA-seq transcript file."""
    filename = os.path.basename(tsv_file)
    sample_id = filename.replace(".tsv.gz", "").replace("-RNA.tsv.gz", "")
    
    # Try to clean up sample ID similarly
    if "_RNA_" in sample_id and "-RNA" in sample_id:
        sample_id = sample_id.split("_RNA_")[1]
        if sample_id.endswith("-RNA"):
            sample_id = sample_id[:-4]
            
    # Or just use filename stem
    sample_id = filename.split('.')[0]
            
    print(f"Processing scRNA-seq: {filename} -> ID: {sample_id}")
    
    out_dir = os.path.join(output_dir_base, "scRNA-seq", sample_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize tracks
    tracks = {}
    for chrom in CHROMOSOMES:
        n_bins = CHROM_SIZES[chrom] // RESOLUTION + 1
        tracks[chrom] = np.zeros(n_bins, dtype=np.float32) # 1D track
        
    # Read TSV
    # Format: hg38_chr1  barcode  ENSG00000...
    opener = gzip.open if str(tsv_file).endswith('.gz') else open
    mode = 'rt' if str(tsv_file).endswith('.gz') else 'r'
    
    count = 0
    missed = 0
    with opener(tsv_file, mode) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            
            # gene_id is usually last
            gene_id = parts[-1]
            if gene_id not in gene_map:
                missed += 1
                continue
                
            chrom, bin_idx = gene_map[gene_id]
            
            if bin_idx < len(tracks[chrom]):
                tracks[chrom][bin_idx] += 1
                
            count += 1
            if count % 1000000 == 0:
                print(f"  Processed {count/1000000:.1f}M transcripts...", end='\r')
                
    print(f"  Total transcripts: {count}, Missed/Unknown Genes: {missed}")
    
    # Save as .npy
    # Note: create_human_npz_v2.py expects {chrom}_{resolution}.npy
    for chrom, track in tracks.items():
        np.save(os.path.join(out_dir, f"{chrom}_{RESOLUTION}.npy"), track)
    print(f"  Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hic-dir', default=DEFAULT_HIC_DIR)
    parser.add_argument('--rna-dir', default=DEFAULT_RNA_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--genes-bed', default=DEFAULT_GENES_BED)
    args = parser.parse_args()
    
    # Load gene map
    gene_map = load_gene_coordinates(args.genes_bed)
    
    # Process Hi-C
    if os.path.exists(args.hic_dir):
        files = list(Path(args.hic_dir).glob("*.pairs.gz"))
        print(f"\nFound {len(files)} Hi-C files.")
        for f in files:
            process_schic_file(f, args.output_dir)
    else:
        print(f"Hi-C directory not found: {args.hic_dir}")
        
    # Process RNA
    if os.path.exists(args.rna_dir):
        files = list(Path(args.rna_dir).glob("*.tsv.gz"))
        print(f"\nFound {len(files)} RNA files.")
        for f in files:
            process_rna_file(f, gene_map, args.output_dir)
    else:
        print(f"RNA directory not found: {args.rna_dir}")
        
    print("\nProcessing complete.")

if __name__ == '__main__':
    main()
