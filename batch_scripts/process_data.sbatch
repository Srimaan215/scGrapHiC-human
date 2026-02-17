#!/bin/bash
#SBATCH --job-name=process_scGrapHiC
#SBATCH --output=/users/ssridh26/jobtmp/process_scgraphic_%j.out
#SBATCH --error=/users/ssridh26/jobtmp/process_scgraphic_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

# =============================================================================
# Process Data Pipeline for Human scGrapHiC
# =============================================================================
# Run this AFTER download_and_prepare_data.sbatch completes
# 
# This script:
# 1. Preprocesses CTCF and CpG annotations
# 2. Creates pseudobulk scHi-C matrices from pairs files
# 3. Creates NPZ files for inference
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "Processing Data for Human scGrapHiC"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Activate conda
source /users/ssridh26/miniforge3/etc/profile.d/conda.sh
conda activate base

# Paths
PROJECT_DIR="/users/ssridh26/projects/t2_human_scgraphic"
DATA_DIR="/users/ssridh26/scratch/human_scGrapHiC"
OUTPUT_DIR="/users/ssridh26/scratch/t2_human_scgraphic"

# Check if downloads completed
if [ ! -d "${DATA_DIR}/GSE238001" ]; then
    echo "ERROR: GSE238001 data not found. Run download_and_prepare_data.sbatch first!"
    exit 1
fi

# Step 1: Preprocess annotations
echo ""
echo "=========================================="
echo "Step 1: Preprocessing CTCF and CpG annotations"
echo "=========================================="

cd ${PROJECT_DIR}
python preprocess_annotations.py \
    --ctcf "${DATA_DIR}/annotations/CTCF_hg38.bed" \
    --cpg "${DATA_DIR}/annotations/cpgIslandExt.txt" \
    --ctcf-out "${DATA_DIR}/annotations/preprocessed/ctcf" \
    --cpg-out "${DATA_DIR}/annotations/preprocessed/cpg" \
    --resolution 50000

# Step 2: Create pseudobulk matrices
echo ""
echo "=========================================="
echo "Step 2: Creating pseudobulk scHi-C matrices"
echo "=========================================="

python create_pseudobulk_v2.py \
    --input-dir "${DATA_DIR}/GSE238001/scHi-C" \
    --output-dir "${DATA_DIR}/GSE238001/pseudobulk_official" \
    --metadata "${DATA_DIR}/GSE238001/metadata_hg38.csv" \
    --resolution 50000

# Step 3: Create NPZ for inference
echo ""
echo "=========================================="
echo "Step 3: Creating NPZ files for inference"
echo "=========================================="

# Check if create_human_npz_v2.py exists
if [ -f "${PROJECT_DIR}/create_human_npz_v2.py" ]; then
    # Process each cell type
    for cell_type in HSC MPP LMPP MEP B_NK; do
        echo "Processing cell type: ${cell_type}"
        python create_human_npz_v2.py \
            --cell_type "${cell_type}" \
            --output_dir "${OUTPUT_DIR}/data"
    done
else
    echo "WARNING: create_human_npz_v2.py not found. Skipping NPZ creation."
    echo "You may need to run NPZ creation manually."
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Data locations:"
echo "  - Preprocessed annotations: ${DATA_DIR}/annotations/preprocessed/"
echo "  - Pseudobulk matrices: ${DATA_DIR}/GSE238001/pseudobulk_official/"
echo "  - NPZ files: ${OUTPUT_DIR}/data/"
echo ""
echo "Next step: Run inference with:"
echo "  cd ${PROJECT_DIR}"
echo "  python inference.py --checkpoint /oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt"
echo ""
echo "Done at $(date)"
