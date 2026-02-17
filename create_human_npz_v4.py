#!/usr/bin/env python3
"""
Create NPZ dataset file for human scGrapHiC inference.

This script prepares the data in the exact format expected by the scGrapHiC model,
using existing preprocessed data from the human_scGrapHiC pipeline.

Data sources:
- scHi-C pseudo-bulk: /users/ssridh26/scratch/human_scGrapHiC/GSE238001/pseudobulk_official/{cell_type}/
- RNA features: /users/ssridh26/scratch/t2_human_scgraphic/GSE238001/scRNA-seq/{cell_type}/
- CTCF: /users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/ctcf/
- CpG: /users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/cpg/
- Bulk Hi-C: /users/ssridh26/scratch/human_scGrapHiC/bulk_hic/ (need to extract)

Key format from training data analysis:
- RNA (ch 0-1): Normalized to [0, 1] using library_size_normalization, shape (2, N)
- CTCF (ch 2-3): Raw accumulated scores, NOT normalized, shape (2, N)
- CpG (ch 4): Raw accumulated scores, NOT normalized, shape (1, N)
- PEs: Shape (N, 128, 16), range [0, 1]
- Targets: Shape (N, 1, 128, 128), range [0, 1]
"""

import numpy as np
import sys
import os
import argparse
from pathlib import Path
from scipy import sparse
from scipy.sparse.csgraph import laplacian
import hicstraw

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths
HUMAN_SCGRAPHIC_DIR = Path("/users/ssridh26/scratch/human_scGrapHiC")
T2_OUTPUT_DIR = Path("/users/ssridh26/scratch/t2_human_scgraphic")

# Data paths - use newly generated data
PSEUDOBULK_GEN_DIR = HUMAN_SCGRAPHIC_DIR / "GSE238001/pseudobulk_generated"
SCHIC_DIR = PSEUDOBULK_GEN_DIR / "scHi-C"
RNA_DIR = PSEUDOBULK_GEN_DIR / "scRNA-seq"
CTCF_DIR = HUMAN_SCGRAPHIC_DIR / "annotations/preprocessed/ctcf"
CPG_DIR = HUMAN_SCGRAPHIC_DIR / "annotations/preprocessed/cpg"
BULK_HIC_FILE = Path("/users/ssridh26/4DNFI5IAH9H1.hic")  # K562 bulk Hi-C from 4DN

# Parameters matching the checkpoint
PARAMETERS = {
    'resolution': 50000,
    'library_size': 25000,
    'normalization_algorithm': 'library_size_normalization',
    'hic_smoothing': True,
    'smoothing_threshold': 0.25,  # Critical: matches training data
    'bounds': 10,
    'stride': 32,
    'padding': True,
    'num_nodes': 128,
    'remove_borders': 30000000,
    'pos_encodings_dim': 16,
    'ctcf_motif': True,
    'cpg_motif': True,
}

# hg38 chromosome sizes
CHROM_SIZES = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
    'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
    'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
    'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
    'chr21': 46709983, 'chr22': 50818468
}

# Cell type mapping for metadata (IDs from dataset_labels.json)
CELL_TYPE_MAP = {
    'HSC': 28, 'MPP': 29, 'LMPP': 30, 'MEP': 31, 'B_NK': 32, 'B-NK': 32,
    'MLP': 29,  # Use MPP ID for MLP
    'Unk_HSC': 33, 'Unk=HSC': 33,
    'Unk_2': 34, 'Unk=2': 34,
    'Unk_5': 35, 'Unk=5': 35,
    'Unk_B_NK': 36, 'Unk=B-NK': 36,
}

# Directory name mapping (for different naming conventions)
CELL_TYPE_DIR_MAP = {
    'B_NK': 'B-NK',
    'Unk_HSC': 'Unk=HSC',
    'Unk_2': 'Unk=2',
    'Unk_5': 'Unk=5',
    'Unk_B_NK': 'Unk=B-NK',
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def library_size_normalization(matrix, library_size):
    """Normalize matrix using library size normalization"""
    matrix = matrix.astype(np.float32)
    mat_sum = matrix.sum()
    if mat_sum > 0:
        matrix = matrix / mat_sum * library_size
    matrix = np.log1p(matrix)
    max_val = matrix.max()
    if max_val > 0:
        matrix = matrix / max_val
    return matrix


def normalize_genomic_track(track, library_size):
    """Normalize genomic track (CTCF, CpG, etc) using library size normalization.
    
    This matches the normalize_track=True setting from training data.
    """
    track = track.astype(np.float32)
    sum_track = np.sum(track)
    if sum_track > 0:
        track = track / sum_track * library_size
    track = np.log1p(track)
    max_val = np.max(track)
    if max_val > 0:
        track = track / max_val
    return track


def soft_thresholding(eigenvalues, threshold):
    """Soft thresholding of eigenvalues"""
    return np.sign(eigenvalues) * np.maximum(np.abs(eigenvalues) - threshold, 0)


def smooth_adjacency_matrix(A, threshold=0.25):
    """
    Smooth adjacency matrix using eigenvalue soft-thresholding.
    
    This is critical for scGrapHiC training data - it spreads signal
    across the matrix and removes zeros, producing dense targets.
    
    Args:
        A: Input adjacency matrix (128, 128)
        threshold: Soft thresholding value (default 0.25 from training)
    
    Returns:
        Smoothed matrix (128, 128)
    """
    # Store diagonal before smoothing
    diagonal = np.diag(A).copy()
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Apply soft thresholding to eigenvalues
    smoothed_eigenvalues = soft_thresholding(eigenvalues, threshold)
    
    # Reconstruct the smoothed adjacency matrix
    smoothed_A = eigenvectors @ np.diag(smoothed_eigenvalues) @ eigenvectors.T
    
    # Ensure real values
    smoothed_A = smoothed_A.real
    
    # Restore diagonal to preserve self-interactions
    np.fill_diagonal(smoothed_A, diagonal)
    
    return smoothed_A


def compactM(matrix, compact_indices):
    """Compact matrix to only include valid rows/columns"""
    return matrix[np.ix_(compact_indices, compact_indices)]


def graph_pe(matrix, pe_dim):
    """Compute positional encodings from Laplacian eigenvectors
    
    Uses abs-max normalization followed by shifting to [0, 1] range,
    matching the original scGrapHiC preprocessing.
    """
    sp_mat = sparse.csr_matrix(matrix.astype(np.float64))
    L = laplacian(sp_mat, normed=True)
    try:
        eigenvalues, eigenvectors = sparse.linalg.eigsh(
            L.astype(np.float64), k=pe_dim+1, which='SM', tol=1e-6
        )
        pe = eigenvectors[:, 1:pe_dim+1]
        
        # Abs-max normalization per eigenvector
        for i in range(pe.shape[1]):
            max_abs = np.max(np.abs(pe[:, i]))
            if max_abs > 1e-12:
                pe[:, i] = pe[:, i] / max_abs
        
        # Shift from [-1, 1] to [0, 1]
        pe = (pe + 1) / 2.0
        
    except Exception as e:
        print(f"    Warning: PE computation failed: {e}")
        pe = np.ones((matrix.shape[0], pe_dim)) * 0.5  # Default to 0.5
    return pe.astype(np.float32)


def divide_matrix(mat, chr_num, params):
    """Divide matrix into overlapping chunks"""
    result = []
    index = []
    size = mat.shape[0]
    
    stride = params['stride']
    chunk_size = params['num_nodes']
    bound = params['bounds']
    padding = params['padding']
    
    if stride < chunk_size and padding:
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
    
    height, width = mat.shape
    
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i-j) <= bound and (i+chunk_size <= height and j+chunk_size <= width):
                subImage = mat[i:i+chunk_size, j:j+chunk_size]
                result.append([subImage])
                index.append((int(chr_num), int(size), int(i), int(j)))  # 4 values like training data
    
    return np.array(result), np.array(index)


def divide_signal(signal, chr_num, params):
    """Divide signal into overlapping chunks"""
    result = []
    
    stride = params['stride']
    chunk_size = params['num_nodes']
    padding = params['padding']
    
    if stride < chunk_size and padding:
        pad_len = (chunk_size - stride) // 2
        signal = np.pad(signal, ((pad_len, pad_len), (0, 0)), 'constant')
    
    size = signal.shape[0]
    
    for i in range(0, size, stride):
        if (i+chunk_size) <= size:
            subImage = signal[i:i+chunk_size, :]
            result.append([subImage])
    
    return np.array(result)


def get_informative_indices(matrix, threshold=0.1):
    """Get indices of rows/columns with sufficient signal"""
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    total = matrix.sum()
    
    if total == 0:
        return np.arange(matrix.shape[0])
    
    # Use bins with non-zero signal
    nonzero_rows = np.where(row_sums > 0)[0]
    nonzero_cols = np.where(col_sums > 0)[0]
    informative = np.intersect1d(nonzero_rows, nonzero_cols)
    
    return informative


def extract_bulk_hic(hic_file, chrom, resolution):
    """Extract bulk Hi-C matrix from .hic file"""
    try:
        # Remove 'chr' prefix for hicstraw
        chrom_name = chrom.replace('chr', '')
        
        hic = hicstraw.HiCFile(str(hic_file))
        
        # Get matrix size
        chrom_size = CHROM_SIZES[chrom]
        n_bins = chrom_size // resolution + 1
        
        # Try to get matrix with NONE normalization (raw counts)
        matrix_zoom = hic.getMatrixZoomData(chrom_name, chrom_name, "observed", "NONE", "BP", resolution)
        
        # Get records as list (safer than getRecordsAsMatrix which can segfault)
        records = matrix_zoom.getRecords(0, chrom_size, 0, chrom_size)
        
        # Convert to dense matrix
        matrix = np.zeros((n_bins, n_bins), dtype=np.float32)
        for record in records:
            bin1 = record.binX // resolution
            bin2 = record.binY // resolution
            if bin1 < n_bins and bin2 < n_bins:
                matrix[bin1, bin2] = record.counts
                matrix[bin2, bin1] = record.counts  # Symmetric
        
        return matrix
        
    except Exception as e:
        print(f"    Warning: Could not extract bulk Hi-C for {chrom}: {e}")
        return None



def get_matching_rna_sample(hic_sample):
    """Find matching RNA sample for a given Hi-C sample"""
    # Logic: GSM7657708_contact_CD-HiC-0722-4.pairs.gz -> GSM7657707_RNA_CD-RNA-0722-4.tsv.gz
    # The IDs in directory names lack .pairs.gz / .tsv.gz
    
    # Try dynamic matching by cleaning up the name
    # Common pattern seems to be swapping 'contact' with 'RNA' and 'HiC' with 'RNA' somewhere
    # Actually, let's list the RNA dir and fuzzy match
    
    if not RNA_DIR.exists():
        print(f"RNA dir not found: {RNA_DIR}")
        return None
        
    rna_samples = [d for d in os.listdir(RNA_DIR) if (RNA_DIR / d).is_dir()]
    
    # Heuristic: Extract the specific part like "CD-HiC-0722-4" -> "CD-RNA-0722-4"
    hic_suffix = hic_sample.split("_contact_")[-1]  # CD-HiC-0722-4
    hic_suffix_clean = hic_suffix.replace("-HiC", "") # CD-0722-4 or similar
    
    # Try to find RNA sample that contains similar suffix
    for rna in rna_samples:
        if hic_suffix_clean in rna:
             return rna
             
    # Fallback: specific replacements seen in logs
    # GSM7657708_contact_CD-HiC-0722-4 -> GSM7657707_RNA_CD-RNA-0722-4
    if "CD-HiC" in hic_sample:
        target = hic_sample.replace("contact", "RNA").replace("CD-HiC", "CD-RNA")
        # The GSM ID also changes (e.g. 708 -> 707), so we can't assume prefix match
        # Just look for the suffix part
        suffix = target.split("_")[-1] # CD-RNA-0722-4
        for rna in rna_samples:
            if suffix in rna:
                return rna
                
    print(f"Could not find RNA match for {hic_sample}")
    return int(0) # Will fail exists check gracefully if we return simple incorrect path or handle None


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_sample(sample_name, output_path):
    """Process a single sample and create NPZ file"""
    
    chromosomes = [f'chr{i}' for i in range(1, 23)]  # Skip chrX
    
    all_node_features = []
    all_targets = []
    all_pes = []
    all_bulk_hics = []
    all_indexes = []
    all_metadatas = []
    
    border_size = PARAMETERS['remove_borders'] // PARAMETERS['resolution']
    print(f"Border size: {border_size} bins")
    
    # Find matching RNA folder
    rna_sample = get_matching_rna_sample(sample_name)
    if not rna_sample:
        print(f"Skipping {sample_name}: No matching RNA found.")
        return
        
    print(f"Processing Pair: {sample_name} <-> {rna_sample}")
    
    for chrom in chromosomes:
        chr_num = int(chrom.replace('chr', ''))
        
        print(f"\nProcessing {chrom}...")
        
        # File paths
        rna_file = RNA_DIR / rna_sample / f"{chrom}_{PARAMETERS['resolution']}.npy"
        schic_file = SCHIC_DIR / sample_name / f"{chrom}.npy"
        ctcf_file = CTCF_DIR / f"{chrom}_{PARAMETERS['resolution']}.npy"
        cpg_file = CPG_DIR / f"{chrom}_{PARAMETERS['resolution']}.npy"
        
        # Check required files exist
        missing_files = []
        if not rna_file.exists():
            missing_files.append(f"RNA: {rna_file}")
        if not schic_file.exists():
            missing_files.append(f"scHi-C: {schic_file}")
        if not ctcf_file.exists():
            missing_files.append(f"CTCF: {ctcf_file}")
        if not cpg_file.exists():
            missing_files.append(f"CpG: {cpg_file}")
        
        if missing_files:
            print(f"  Skipping - missing files:")
            for mf in missing_files:
                print(f"    - {mf}")
            continue
        
        # Load scHi-C (target)
        schic_data = np.load(schic_file).astype(np.float32)
        print(f"  scHi-C shape: {schic_data.shape}")
        
        # Extract bulk Hi-C first
        bulk_hic = extract_bulk_hic(BULK_HIC_FILE, chrom, PARAMETERS['resolution'])
        if bulk_hic is None:
            print(f"  Skipping - could not load bulk Hi-C")
            continue
        
        # Ensure consistent size
        min_size = min(schic_data.shape[0], bulk_hic.shape[0])
        schic_data = schic_data[:min_size, :min_size]
        bulk_hic = bulk_hic[:min_size, :min_size]
        
        # Get informative indices from BULK Hi-C (not scHi-C)
        # This matches original pipeline which uses bulk's compact indices
        informative_indexes = get_informative_indices(bulk_hic)
        print(f"  Informative bins (from bulk): {len(informative_indexes)}")
        
        if len(informative_indexes) < PARAMETERS['num_nodes']:
            print(f"  Skipping - too few informative bins")
            continue
            
        informative_indexes = informative_indexes[informative_indexes < min_size]
        
        # Compact matrices
        schic_compact = compactM(schic_data, informative_indexes)
        bulk_compact = compactM(bulk_hic, informative_indexes)
        
        # Remove borders
        if schic_compact.shape[0] <= border_size:
            print(f"  Skipping - too small after compaction: {schic_compact.shape[0]}")
            continue
            
        schic_compact = schic_compact[border_size:, border_size:]
        bulk_compact = bulk_compact[border_size:, border_size:]
        
        if schic_compact.shape[0] < PARAMETERS['num_nodes']:
            print(f"  Skipping - too small after border removal: {schic_compact.shape[0]}")
            continue
        
        # Divide matrices
        schic_chunks, indexes = divide_matrix(schic_compact, chr_num, PARAMETERS)
        bulk_chunks, _ = divide_matrix(bulk_compact, chr_num, PARAMETERS)
        
        if len(schic_chunks) == 0:
            print(f"  Skipping - no valid chunks")
            continue
        
        # Load and process features
        rna_data = np.load(rna_file)
        # Ensure RNA is 2D with 2 channels
        if rna_data.ndim == 1:
            # Replicate 1D RNA to 2 channels
            rna_data = np.stack([rna_data, rna_data], axis=0)
        elif rna_data.shape[0] != 2 and rna_data.shape[1] == 2:
            rna_data = rna_data.T
            
        ctcf_data = np.load(ctcf_file)  # (2, N)
        cpg_data = np.load(cpg_file)  # (1, N)
        
        print(f"  Feature shapes - RNA: {rna_data.shape}, CTCF: {ctcf_data.shape}, CpG: {cpg_data.shape}")
        
        # Normalize RNA (library size)
        rna_data = np.array([
            normalize_genomic_track(rna_data[0], PARAMETERS['library_size']),
            normalize_genomic_track(rna_data[1], PARAMETERS['library_size'])
        ])
        
        # Normalize CTCF and CpG features (matching training data preprocessing)
        # Training data uses normalize_track=True which applies library_size normalization
        ctcf_data = np.array([
            normalize_genomic_track(ctcf_data[0], PARAMETERS['library_size']),
            normalize_genomic_track(ctcf_data[1], PARAMETERS['library_size'])
        ])
        cpg_data = np.array([normalize_genomic_track(cpg_data[0], PARAMETERS['library_size'])])
        
        print(f"  After normalization - CTCF: [{ctcf_data.min():.3f}, {ctcf_data.max():.3f}], CpG: [{cpg_data.min():.3f}, {cpg_data.max():.3f}]")
        
        # Handle size mismatch with informative indices
        max_idx = max(informative_indexes) if len(informative_indexes) > 0 else 0
        
        for name, data in [("RNA", rna_data), ("CTCF", ctcf_data), ("CpG", cpg_data)]:
            if max_idx >= data.shape[1]:
                print(f"  Warning: {name} needs padding from {data.shape[1]} to {max_idx+1}")
        
        # Pad features if needed
        if max_idx >= rna_data.shape[1]:
            pad_size = max_idx - rna_data.shape[1] + 1
            rna_data = np.pad(rna_data, ((0, 0), (0, pad_size)), mode='constant')
        if max_idx >= ctcf_data.shape[1]:
            pad_size = max_idx - ctcf_data.shape[1] + 1
            ctcf_data = np.pad(ctcf_data, ((0, 0), (0, pad_size)), mode='constant')
        if max_idx >= cpg_data.shape[1]:
            pad_size = max_idx - cpg_data.shape[1] + 1
            cpg_data = np.pad(cpg_data, ((0, 0), (0, pad_size)), mode='constant')
        
        # Take informative indices
        rna_compact = rna_data.take(informative_indexes, axis=1)
        ctcf_compact = ctcf_data.take(informative_indexes, axis=1)
        cpg_compact = cpg_data.take(informative_indexes, axis=1)
        
        # Remove borders
        rna_compact = rna_compact[:, border_size:]
        ctcf_compact = ctcf_compact[:, border_size:]
        cpg_compact = cpg_compact[:, border_size:]
        
        # Concatenate features: (5, N) = 2 RNA + 2 CTCF + 1 CpG
        node_features = np.concatenate([rna_compact, ctcf_compact, cpg_compact], axis=0)
        
        # Divide features
        features_chunks = divide_signal(node_features.T, chr_num, PARAMETERS)
        features_chunks = features_chunks[:, 0, :, :]  # (num_chunks, 128, 5)
        
        # Ensure same number of chunks
        n_chunks = min(len(schic_chunks), len(bulk_chunks), len(features_chunks))
        schic_chunks = schic_chunks[:n_chunks]
        bulk_chunks = bulk_chunks[:n_chunks]
        features_chunks = features_chunks[:n_chunks]
        indexes = indexes[:n_chunks]
        
        # Compute positional encodings for each chunk
        pes = []
        for i in range(n_chunks):
            chunk = bulk_chunks[i, 0, :, :]
            pe = graph_pe(chunk, PARAMETERS['pos_encodings_dim'])
            pes.append(pe)
        pes = np.array(pes)
        
        # Normalize Hi-C matrices and apply smoothing to targets
        # Smoothing is critical - it spreads signal and removes zeros
        schic_norm = []
        bulk_norm = []
        for i in range(n_chunks):
            sc = library_size_normalization(schic_chunks[i, 0], PARAMETERS['library_size'])
            bl = library_size_normalization(bulk_chunks[i, 0], PARAMETERS['library_size'])
            
            # Apply eigenvalue soft-thresholding smoothing to scHi-C (target)
            # This matches training data preprocessing: hic_smoothing=True, smoothing_threshold=0.25
            sc = smooth_adjacency_matrix(sc, threshold=PARAMETERS['smoothing_threshold'])
            
            schic_norm.append(sc)
            bulk_norm.append(bl)
        
        schic_norm = np.array(schic_norm)[:, np.newaxis, :, :]
        bulk_norm = np.array(bulk_norm)[:, np.newaxis, :, :]
        

        # Create metadata: [stage, tissue, cell_type, sample_id]
        # From dataset_labels.json: human_blood=8, blood=2, HSC=28, etc.
        metadatas = np.zeros((n_chunks, 4), dtype=np.int32)
        metadatas[:, 0] = 8  # stage = human_blood
        metadatas[:, 1] = 2  # tissue = blood
        # Infer cell type from sample name if possible, else 28 (HSC)
        ct_id = 28
        for ct_name, ct_val in CELL_TYPE_MAP.items():
            if ct_name in sample_name:
                ct_id = ct_val
                break
        
        metadatas[:, 2] = ct_id  # cell type ID
        metadatas[:, 3] = chr_num  # chromosome as sample identifier
        
        print(f"  Created {n_chunks} chunks")
        
        # Append to lists
        all_node_features.append(features_chunks)
        all_targets.append(schic_norm)
        all_pes.append(pes)
        all_bulk_hics.append(bulk_norm)
        all_indexes.append(indexes)
        all_metadatas.append(metadatas)
    
    if not all_node_features:
        print("ERROR: No valid data created!")
        return
    
    # Concatenate all chromosomes
    node_features = np.concatenate(all_node_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    pes = np.concatenate(all_pes, axis=0)
    bulk_hics = np.concatenate(all_bulk_hics, axis=0)
    indexes = np.concatenate(all_indexes, axis=0)
    metadatas = np.concatenate(all_metadatas, axis=0)
    
    print("\n" + "=" * 60)
    print(f"Final shapes:")
    print(f"  node_features: {node_features.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  pes: {pes.shape}")
    print(f"  bulk_hics: {bulk_hics.shape}")
    print(f"  indexes: {indexes.shape}")
    print(f"  metadatas: {metadatas.shape}")
    
    # Print feature statistics
    print("\nFeature statistics:")
    for i in range(node_features.shape[2]):
        vals = node_features[..., i]
        print(f"  Ch {i}: [{vals.min():.4f}, {vals.max():.4f}], mean={vals.mean():.4f}")
    
    # Save NPZ
    output_file = output_path / f"{sample_name}_inference.npz"
    np.savez(
        output_file,
        node_features=node_features,
        targets=targets,
        pes=pes,
        bulk_hics=bulk_hics,
        indexes=indexes,
        metadatas=metadatas
    )
    
    print(f"\nSaved to: {output_file}")
    print(f"Total samples: {node_features.shape[0]}")


def main():
    parser = argparse.ArgumentParser(description='Create NPZ dataset for human scGrapHiC inference')
    parser.add_argument('--samples', type=str, nargs='+', 
                        help='Specific samples to process (names of folders in scHi-C dir)')
    parser.add_argument('--output_dir', type=str, 
                        default=str(T2_OUTPUT_DIR / "processed/generated_npz"),
                        help='Output directory for NPZ files')
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.samples:
        samples = args.samples
    else:
        # Default scan directory
        if not SCHIC_DIR.exists():
            print(f"scHi-C directory not found: {SCHIC_DIR}")
            return
        samples = [d for d in os.listdir(SCHIC_DIR) if (SCHIC_DIR / d).is_dir()]
        
    print(f"Processing {len(samples)} samples...")
        
    for sample in samples:
        print("=" * 60)
        print(f"Creating NPZ for: {sample}")
        print("=" * 60)
        try:
            process_sample(sample, output_path)
        except Exception as e:
            print(f"Error processing {sample}: {e}")


if __name__ == "__main__":
    main()
