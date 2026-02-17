#!/bin/bash
#SBATCH --job-name=pseudobulk
#SBATCH --output=/users/ssridh26/jobtmp/pseudobulk_%j.out
#SBATCH --error=/users/ssridh26/jobtmp/pseudobulk_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# =============================================================================
# Create Pseudobulk scHi-C matrices from GAGE-seq pairs files
# =============================================================================

set -e

echo "=========================================="
echo "Creating Pseudobulk scHi-C Matrices"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Activate conda
source /users/ssridh26/miniforge3/etc/profile.d/conda.sh
conda activate base

cd /users/ssridh26/projects/t2_human_scgraphic

python -u create_pseudobulk_v2.py \
    --input-dir /users/ssridh26/scratch/human_scGrapHiC/GSE238001/scHi-C \
    --output-dir /users/ssridh26/scratch/human_scGrapHiC/GSE238001/pseudobulk_official \
    --metadata /users/ssridh26/scratch/human_scGrapHiC/GSE238001/metadata_hg38.csv \
    --resolution 50000

echo ""
echo "Done at $(date)"
