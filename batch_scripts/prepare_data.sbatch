#!/bin/bash
#SBATCH --job-name=prepare_human_data
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/prepare_data_%j.log
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/prepare_data_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Prepare human RNA features with proper library_size normalization

echo "=========================================="
echo "Preparing Human Data for scGrapHiC"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Activate environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

# Run preparation script
python prepare_human_data.py

echo ""
echo "End time: $(date)"
echo "=========================================="
