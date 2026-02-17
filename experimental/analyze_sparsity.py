
import os
import numpy as np
from pathlib import Path
import glob

# Base paths
HUMAN_SCGRAPHIC_DIR = Path("/users/ssridh26/scratch/human_scGrapHiC")
PSEUDOBULK_GEN_DIR = HUMAN_SCGRAPHIC_DIR / "GSE238001/pseudobulk_generated"
SCHIC_BASE_DIR = PSEUDOBULK_GEN_DIR / "scHi-C"

def calculate_sparsity(matrix):
    total_elements = matrix.size
    if total_elements == 0:
        return 1.0
    zero_elements = total_elements - np.count_nonzero(matrix)
    return zero_elements / total_elements

def analyze_samples():
    if not SCHIC_BASE_DIR.exists():
        print(f"Directory not found: {SCHIC_BASE_DIR}")
        return

    samples = [d for d in os.listdir(SCHIC_BASE_DIR) if os.path.isdir(SCHIC_BASE_DIR / d)]
    print(f"Found {len(samples)} samples.")
    
    sample_stats = []

    for sample in samples:
        sample_dir = SCHIC_BASE_DIR / sample
        chrom_files = glob.glob(str(sample_dir / "chr*.npy"))
        
        total_sparsity = 0
        file_count = 0
        total_reads = 0
        
        for f in chrom_files:
            try:
                data = np.load(f)
                sp = calculate_sparsity(data)
                total_sparsity += sp
                total_reads += np.sum(data)
                file_count += 1
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if file_count > 0:
            avg_sparsity = total_sparsity / file_count
            sample_stats.append({
                'name': sample,
                'sparsity': avg_sparsity,
                'reads': total_reads
            })
            print(f"Sample: {sample}, Avg Sparsity: {avg_sparsity:.4f}, Total Reads: {total_reads}")
        else:
            print(f"Sample: {sample} has no valid chromosome files.")

    # Sort by sparsity (ascending - lowest sparsity means most dense/best data)
    sample_stats.sort(key=lambda x: x['sparsity'])
    
    print("\n--- Top 5 Lowest Sparsity Samples (Best Quality) ---")
    for i, s in enumerate(sample_stats[:5]):
        print(f"{i+1}. {s['name']}: {s['sparsity']:.4f} ({s['sparsity']*100:.2f}%) - Reads: {s['reads']}")

    print("\n--- Bottom 5 Highest Sparsity Samples (Worst Quality) ---")
    for i, s in enumerate(sample_stats[-5:]):
        print(f"{i+1}. {s['name']}: {s['sparsity']:.4f} ({s['sparsity']*100:.2f}%) - Reads: {s['reads']}")

if __name__ == "__main__":
    analyze_samples()
