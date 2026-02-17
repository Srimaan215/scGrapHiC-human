#!/usr/bin/env python3
"""
Create NPZ dataset file for human scGrapHiC inference.
This script prepares the data in the exact format expected by the scGrapHiC model.
"""

import numpy as np
import sys
import argparse
from pathlib import Path
from scipy import sparse
from scipy.sparse.csgraph import laplacian

# Parameters matching the checkpoint
PARAMETERS = {
    'resolution': 50000,
    'library_size': 25000,
    'normalization_algorithm': 'library_size_normalization',
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
    'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895
}

# Cell type mapping
CELL_TYPE_MAP = {'HSC': 28, 'MPP': 29, 'LMPP': 30, 'MEP': 31, 'B_NK': 32}


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


def compactM(matrix, compact_indices):
    """Compact matrix to only include valid rows/columns"""
    return matrix[np.ix_(compact_indices, compact_indices)]


def graph_pe(matrix, pe_dim):
    """Compute positional encodings from Laplacian eigenvectors"""
    sp_mat = sparse.csr_matrix(matrix.astype(np.float64))
    L = laplacian(sp_mat, normed=True)
    try:
        eigenvalues, eigenvectors = sparse.linalg.eigsh(
            L.astype(np.float64), k=pe_dim+1, which='SM', tol=1e-6
        )
        pe = eigenvectors[:, 1:pe_dim+1]
    except:
        pe = np.zeros((matrix.shape[0], pe_dim))
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
            if abs(i-j) <= bound and (i+chunk_size < height and j+chunk_size < width):
                subImage = mat[i:i+chunk_size, j:j+chunk_size]
                result.append([subImage])
                index.append((int(chr_num), int(size), int(i)))
    
    return np.array(result), np.array(index)


def divide_signal(signal, chr_num, params):
    """Divide signal into overlapping chunks"""
    result = []
    index = []
    
    stride = params['stride']
    chunk_size = params['num_nodes']
    padding = params['padding']
    
    if stride < chunk_size and padding:
        pad_len = (chunk_size - stride) // 2
        signal = np.pad(signal, ((pad_len, pad_len), (0, 0)), 'constant')
    
    size = signal.shape[0]
    
    for i in range(0, size, stride):
        if (i+chunk_size) < size:
            subImage = signal[i:i+chunk_size, :]
            result.append([subImage])
            index.append((int(chr_num), int(size), int(i)))
    
    return np.array(result), np.array(index)


def process_cell_type(cell_type, base_path, output_path):
    """Process a single cell type and create NPZ file"""
    
    chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    
    all_node_features = []
    all_targets = []
    all_pes = []
    all_bulk_hics = []
    all_indexes = []
    all_metadatas = []
    
    border_size = PARAMETERS['remove_borders'] // PARAMETERS['resolution']
    print(f"Border size: {border_size} bins")
    
    for chrom in chromosomes:
        chr_num = chrom.replace('chr', '')
        if chr_num == 'X':
            chr_num = 23
        else:
            chr_num = int(chr_num)
        
        print(f"\nProcessing {chrom}...")
        
        # File paths
        rna_file = base_path / f"GSE238001/scRNA-seq/{cell_type}_scrnaseq/{chrom}_{PARAMETERS['resolution']}.npy"
        schic_file = base_path / f"GSE238001/pseudo-bulk/scHi-C/{cell_type}_schic/{chrom}_{PARAMETERS['resolution']}.npy"
        bulk_file = base_path / f"bulk/K562/{chrom}_{PARAMETERS['resolution']}.npz"
        # Use preprocessed motifs from previous pipeline (NOT normalized to [0,1])
        ctcf_file = Path("/users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/ctcf") / f"{chrom}_{PARAMETERS['resolution']}.npy"
        cpg_file = Path("/users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/cpg") / f"{chrom}_{PARAMETERS['resolution']}.npy"
        
        if not all(f.exists() for f in [rna_file, schic_file, bulk_file, ctcf_file, cpg_file]):
            print(f"  Skipping - missing files")
            continue
        
        # Load bulk Hi-C for informative indices
        bulk_obj = np.load(bulk_file)
        informative_indexes = bulk_obj['compact']
        bulk_hic = bulk_obj['hic'].astype(np.float32)
        
        # Load scHi-C
        schic_data = np.load(schic_file).astype(np.float32)
        
        # Check dimensions match
        if schic_data.shape[0] != bulk_hic.shape[0]:
            print(f"  Size mismatch: scHi-C={schic_data.shape[0]}, bulk={bulk_hic.shape[0]}")
            continue
        
        # Compact matrices
        schic_compact = compactM(schic_data, informative_indexes)
        bulk_compact = compactM(bulk_hic, informative_indexes)
        
        # Remove borders
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
        rna_data = np.load(rna_file)  # (2, N)
        ctcf_data = np.load(ctcf_file)  # (2, N)
        cpg_data = np.load(cpg_file)  # (1, N)
        
        # Handle size mismatch with informative indices
        max_idx = max(informative_indexes)
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
        rna_data = rna_data.take(informative_indexes, axis=1)
        ctcf_data = ctcf_data.take(informative_indexes, axis=1)
        cpg_data = cpg_data.take(informative_indexes, axis=1)
        
        # Remove borders
        rna_data = rna_data[:, border_size:]
        ctcf_data = ctcf_data[:, border_size:]
        cpg_data = cpg_data[:, border_size:]
        
        # Concatenate features: (5, N) = 2 RNA + 2 CTCF + 1 CpG
        node_features = np.concatenate([rna_data, ctcf_data, cpg_data], axis=0)
        
        # Divide features
        features_chunks, _ = divide_signal(node_features.T, chr_num, PARAMETERS)
        features_chunks = features_chunks[:, 0, :, :]  # (num_chunks, 128, 5)
        
        # Compute positional encodings for each chunk
        pes = []
        for i in range(len(bulk_chunks)):
            chunk = bulk_chunks[i, 0, :, :]
            pe = graph_pe(chunk, PARAMETERS['pos_encodings_dim'])
            pes.append(pe)
        pes = np.array(pes)
        
        # Normalize Hi-C matrices
        schic_norm = []
        bulk_norm = []
        for i in range(len(schic_chunks)):
            sc = library_size_normalization(schic_chunks[i, 0], PARAMETERS['library_size'])
            bl = library_size_normalization(bulk_chunks[i, 0], PARAMETERS['library_size'])
            schic_norm.append(sc)
            bulk_norm.append(bl)
        
        schic_norm = np.array(schic_norm)[:, np.newaxis, :, :]
        bulk_norm = np.array(bulk_norm)[:, np.newaxis, :, :]
        
        # Create metadata
        num_chunks = len(schic_chunks)
        metadatas = np.zeros((num_chunks, 4), dtype=np.int32)
        metadatas[:, 2] = CELL_TYPE_MAP[cell_type]
        metadatas[:, 3] = 50  # num_cells placeholder
        
        # Append to lists
        all_node_features.append(features_chunks)
        all_targets.append(schic_norm)
        all_pes.append(pes)
        all_bulk_hics.append(bulk_norm)
        all_indexes.append(indexes)
        all_metadatas.append(metadatas)
        
        print(f"  Added {num_chunks} chunks")
    
    # Concatenate all
    print("\n\nConcatenating all data...")
    node_features = np.concatenate(all_node_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    pes = np.concatenate(all_pes, axis=0)
    bulk_hics = np.concatenate(all_bulk_hics, axis=0)
    indexes = np.concatenate(all_indexes, axis=0)
    metadatas = np.concatenate(all_metadatas, axis=0)
    
    print(f"\nFinal shapes:")
    print(f"  node_features: {node_features.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  pes: {pes.shape}")
    print(f"  bulk_hics: {bulk_hics.shape}")
    print(f"  indexes: {indexes.shape}")
    print(f"  metadatas: {metadatas.shape}")
    
    # Save
    output_file = output_path / f"{cell_type}_inference.npz"
    np.savez(output_file,
             node_features=node_features,
             targets=targets,
             pes=pes,
             bulk_hics=bulk_hics,
             indexes=indexes,
             metadatas=metadatas)
    
    print(f"\nSaved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Create NPZ for human scGrapHiC inference')
    parser.add_argument('--cell-type', type=str, default='HSC',
                        choices=['HSC', 'MPP', 'LMPP', 'MEP', 'B_NK'],
                        help='Cell type to process')
    parser.add_argument('--base-path', type=str, 
                        default='/users/ssridh26/scratch/t2_human_scgraphic/preprocessed/hg38',
                        help='Base path for preprocessed data')
    parser.add_argument('--output-path', type=str,
                        default='/users/ssridh26/scratch/t2_human_scgraphic/processed/hg38/GSE238001',
                        help='Output path for NPZ file')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {args.cell_type}...")
    print(f"Base path: {base_path}")
    print(f"Output path: {output_path}")
    
    process_cell_type(args.cell_type, base_path, output_path)


if __name__ == '__main__':
    main()
