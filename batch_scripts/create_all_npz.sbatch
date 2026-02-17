#!/bin/bash
#SBATCH --job-name=create_all_npz
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_all_npz_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_all_npz_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Create logs directory if it doesn't exist
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

# Change to project directory
cd /users/ssridh26/projects/t2_human_scgraphic

echo "============================================"
echo "Creating NPZ for all cell types"
echo "Start time: $(date)"
echo "============================================"

# Process all cell types
for CELL_TYPE in HSC MPP LMPP MEP B_NK; do
    echo ""
    echo "Processing ${CELL_TYPE}..."
    echo "--------------------------------------------"
    
    python create_human_npz.py \
        --cell-type ${CELL_TYPE} \
        --base-path /users/ssridh26/scratch/t2_human_scgraphic/preprocessed/hg38 \
        --output-path /users/ssridh26/scratch/t2_human_scgraphic/processed/hg38/GSE238001
    
    echo "Completed ${CELL_TYPE}"
done

echo "============================================"
echo "All cell types processed!"
echo "End time: $(date)"
echo "============================================"
