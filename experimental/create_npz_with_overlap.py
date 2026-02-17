#!/usr/bin/env python3
"""
Create NPZ files with adjustable window overlap for data augmentation.

This is a modified version of create_human_npz_v2.py that allows control
over window overlap through the --window_overlap parameter.

Usage:
    # No overlap (default behavior)
    python create_npz_with_overlap.py --cell_type HSC --window_overlap 0
    
    # 50% overlap (2x data)
    python create_npz_with_overlap.py --cell_type HSC --window_overlap 0.5
    
    # 75% overlap (4x data)
    python create_npz_with_overlap.py --cell_type HSC --window_overlap 0.75
"""

import sys
import argparse
from pathlib import Path

# Import from existing script
sys.path.insert(0, str(Path(__file__).parent))
from create_human_npz_v2 import (
    PARAMETERS, CHROM_SIZES, CELL_TYPE_MAP, CELL_TYPE_DIR_MAP,
    library_size_normalization, smooth_hic, graph_pe,
    divide_image, divide_signal, get_informative_indices,
    load_bulk_hic, process_cell_type as original_process_cell_type,
    HUMAN_SCGRAPHIC_DIR, T2_OUTPUT_DIR,
    SCHIC_DIR, CTCF_DIR, CPG_DIR, RNA_DIR, BULK_HIC_FILE
)
import numpy as np


def process_cell_type_with_overlap(cell_type, output_path, window_overlap=0.0):
    """
    Process cell type with configurable window overlap.
    
    Args:
        cell_type: Cell type name
        output_path: Output directory
        window_overlap: Overlap fraction (0.0 = no overlap, 0.5 = 50%, 0.75 = 75%)
    """
    # Modify parameters for this run
    params = PARAMETERS.copy()
    
    # Calculate stride from overlap
    # overlap = 1 - (stride / window_size)
    # stride = window_size * (1 - overlap)
    window_size = params['num_nodes']  # 128
    params['stride'] = int(window_size * (1 - window_overlap))
    
    print(f"Window overlap configuration:")
    print(f"  Window size: {window_size}")
    print(f"  Overlap: {window_overlap*100:.0f}%")
    print(f"  Stride: {params['stride']}")
    print()
    
    # Use modified divide functions
    original_divide_image_fn = divide_image
    original_divide_signal_fn = divide_signal
    
    def divide_image_overlap(contact_map, chr_num, params_local):
        """Modified divide_image using custom params"""
        result = []
        index = []
        
        stride = params['stride']  # Use modified stride
        chunk_size = params_local['num_nodes']
        padding = params_local['padding']
        
        if stride < chunk_size and padding:
            pad_len = (chunk_size - stride) // 2
            contact_map = np.pad(contact_map, ((pad_len, pad_len), (0, 0)), 'constant')
        
        height, width = contact_map.shape[0], contact_map.shape[1]
        
        for i in range(0, height, stride):
            for j in range(0, width, stride):
                if (i+chunk_size) <= height and (j+chunk_size) <= width:
                    subImage = contact_map[i:i+chunk_size, j:j+chunk_size]
                    result.append([subImage])
                    index.append((int(chr_num), int(height), int(i), int(j)))
        
        return np.array(result), np.array(index)
    
    def divide_signal_overlap(signal, chr_num, params_local):
        """Modified divide_signal using custom params"""
        result = []
        
        stride = params['stride']  # Use modified stride
        chunk_size = params_local['num_nodes']
        padding = params_local['padding']
        
        if stride < chunk_size and padding:
            pad_len = (chunk_size - stride) // 2
            signal = np.pad(signal, ((pad_len, pad_len), (0, 0)), 'constant')
        
        size = signal.shape[0]
        
        for i in range(0, size, stride):
            if (i+chunk_size) <= size:
                subImage = signal[i:i+chunk_size, :]
                result.append([subImage])
        
        return np.array(result)
    
    # Temporarily replace functions
    import create_human_npz_v2
    create_human_npz_v2.divide_image = divide_image_overlap
    create_human_npz_v2.divide_signal = divide_signal_overlap
    create_human_npz_v2.PARAMETERS = params
    
    # Process using modified parameters
    try:
        original_process_cell_type(cell_type, output_path)
    finally:
        # Restore original functions
        create_human_npz_v2.divide_image = original_divide_image_fn
        create_human_npz_v2.divide_signal = original_divide_signal_fn
        create_human_npz_v2.PARAMETERS = PARAMETERS


def main():
    parser = argparse.ArgumentParser(
        description='Create NPZ dataset with configurable window overlap'
    )
    parser.add_argument('--cell_type', type=str, required=True,
                        help='Cell type to process (HSC, MPP, LMPP, MEP, B_NK)')
    parser.add_argument('--output_dir', type=str, 
                        default=str(T2_OUTPUT_DIR / "processed"),
                        help='Output directory for NPZ files')
    parser.add_argument('--window_overlap', type=float, default=0.0,
                        help='Window overlap fraction (0.0=none, 0.5=50%%, 0.75=75%%)')
    
    args = parser.parse_args()
    
    if not (0 <= args.window_overlap < 1.0):
        raise ValueError("window_overlap must be in range [0, 1)")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Creating NPZ with {args.window_overlap*100:.0f}% window overlap")
    print(f"Cell type: {args.cell_type}")
    print("=" * 70)
    print()
    
    process_cell_type_with_overlap(args.cell_type, output_path, args.window_overlap)
    
    # Rename output file to indicate overlap
    if args.window_overlap > 0:
        original_file = output_path / f"{args.cell_type}_inference.npz"
        overlap_pct = int(args.window_overlap * 100)
        new_file = output_path / f"{args.cell_type}_overlap{overlap_pct}.npz"
        if original_file.exists():
            original_file.rename(new_file)
            print(f"\nRenamed to: {new_file}")


if __name__ == "__main__":
    main()
