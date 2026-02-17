#!/usr/bin/env python3
"""
Preprocess CTCF and CpG annotations for scGrapHiC.

This script:
1. Reads CTCF bigWig file and creates per-chromosome .npy files with scores binned at 50kb
2. Reads CpG Islands BED file and creates per-chromosome .npy files with density binned at 50kb

Output format matches what scGrapHiC expects:
- CTCF: (n_bins,) array of accumulated ChIP-seq signal per bin
- CpG: (n_bins,) array of CpG density per bin
"""

import numpy as np
import os
import argparse
from pathlib import Path

# Try to import pyBigWig, fall back to alternative if not available
try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False
    print("Warning: pyBigWig not available. Install with: pip install pyBigWig")

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


def preprocess_ctcf_bed(bed_path, output_dir, resolution=50000):
    """
    Convert CTCF BED peaks to per-chromosome numpy arrays.
    
    The BED format from ENCODE clustered TF binding:
    chrom, start, end, name(CTCF), score
    
    We accumulate the peak scores per genomic bin.
    
    Args:
        bed_path: Path to CTCF BED file
        output_dir: Directory to save .npy files
        resolution: Bin size in bp (default 50kb)
    """
    print(f"Processing CTCF BED peaks: {bed_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize arrays for all chromosomes
    ctcf_data = {}
    for chrom in CHROMOSOMES:
        chrom_size = CHROM_SIZES.get(chrom)
        if chrom_size:
            n_bins = chrom_size // resolution + 1
            ctcf_data[chrom] = np.zeros(n_bins, dtype=np.float32)
    
    # Parse CTCF BED file
    with open(bed_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            
            chrom = parts[0]
            if chrom not in ctcf_data:
                continue
            
            try:
                start = int(parts[1])
                end = int(parts[2])
                score = float(parts[4])  # Peak score
                
                # Distribute score across overlapping bins
                start_bin = start // resolution
                end_bin = end // resolution
                
                for bin_idx in range(start_bin, min(end_bin + 1, len(ctcf_data[chrom]))):
                    # Add fraction of score that falls in this bin
                    bin_start = bin_idx * resolution
                    bin_end = (bin_idx + 1) * resolution
                    
                    overlap_start = max(start, bin_start)
                    overlap_end = min(end, bin_end)
                    peak_len = end - start if end > start else 1
                    overlap_frac = (overlap_end - overlap_start) / peak_len
                    
                    ctcf_data[chrom][bin_idx] += score * overlap_frac
            except (ValueError, IndexError):
                continue
    
    # Save arrays
    for chrom, scores in ctcf_data.items():
        output_path = os.path.join(output_dir, f"{chrom}.npy")
        np.save(output_path, scores)
        print(f"  {chrom}: {len(scores)} bins, max={scores.max():.2f}, saved to {output_path}")
    
    print("CTCF processing complete!")


def preprocess_cpg_bed(cpg_bed_path, output_dir, resolution=50000):
    """
    Convert CpG Islands BED to per-chromosome numpy arrays.
    
    The UCSC cpgIslandExt.txt format:
    bin, chrom, chromStart, chromEnd, name, length, cpgNum, gcNum, perCpg, perGc, obsExp
    
    We accumulate CpG count (cpgNum) per genomic bin.
    
    Args:
        cpg_bed_path: Path to cpgIslandExt.txt
        output_dir: Directory to save .npy files
        resolution: Bin size in bp (default 50kb)
    """
    print(f"Processing CpG Islands: {cpg_bed_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize arrays for all chromosomes
    cpg_data = {}
    for chrom in CHROMOSOMES:
        chrom_size = CHROM_SIZES.get(chrom)
        if chrom_size:
            n_bins = chrom_size // resolution + 1
            cpg_data[chrom] = np.zeros(n_bins, dtype=np.float32)
    
    # Parse CpG Islands file
    with open(cpg_bed_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            
            chrom = parts[1]
            if chrom not in cpg_data:
                continue
            
            try:
                start = int(parts[2])
                end = int(parts[3])
                cpg_num = float(parts[6])  # Number of CpGs in island
                
                # Distribute CpG count across overlapping bins
                start_bin = start // resolution
                end_bin = end // resolution
                
                for bin_idx in range(start_bin, min(end_bin + 1, len(cpg_data[chrom]))):
                    # Add fraction of CpGs that fall in this bin
                    bin_start = bin_idx * resolution
                    bin_end = (bin_idx + 1) * resolution
                    
                    overlap_start = max(start, bin_start)
                    overlap_end = min(end, bin_end)
                    overlap_frac = (overlap_end - overlap_start) / (end - start) if end > start else 0
                    
                    cpg_data[chrom][bin_idx] += cpg_num * overlap_frac
            except (ValueError, IndexError):
                continue
    
    # Save arrays
    for chrom, scores in cpg_data.items():
        output_path = os.path.join(output_dir, f"{chrom}.npy")
        np.save(output_path, scores)
        print(f"  {chrom}: {len(scores)} bins, max={scores.max():.2f}, saved to {output_path}")
    
    print("CpG processing complete!")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CTCF and CpG annotations')
    parser.add_argument('--ctcf', type=str, 
                        default='/users/ssridh26/scratch/human_scGrapHiC/annotations/CTCF_hg38.bed',
                        help='Path to CTCF BED file')
    parser.add_argument('--cpg', type=str,
                        default='/users/ssridh26/scratch/human_scGrapHiC/annotations/cpgIslandExt.txt',
                        help='Path to CpG Islands BED file')
    parser.add_argument('--ctcf-out', type=str,
                        default='/users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/ctcf',
                        help='Output directory for CTCF .npy files')
    parser.add_argument('--cpg-out', type=str,
                        default='/users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/cpg',
                        help='Output directory for CpG .npy files')
    parser.add_argument('--resolution', type=int, default=50000,
                        help='Bin resolution in bp')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Preprocessing CTCF and CpG Annotations for scGrapHiC")
    print("=" * 60)
    print(f"Resolution: {args.resolution} bp")
    print()
    
    # Process CTCF
    if os.path.exists(args.ctcf):
        preprocess_ctcf_bed(args.ctcf, args.ctcf_out, args.resolution)
    else:
        print(f"WARNING: CTCF file not found: {args.ctcf}")
    
    print()
    
    # Process CpG
    if os.path.exists(args.cpg):
        preprocess_cpg_bed(args.cpg, args.cpg_out, args.resolution)
    else:
        print(f"WARNING: CpG file not found: {args.cpg}")
    
    print()
    print("Done!")


if __name__ == '__main__':
    main()
