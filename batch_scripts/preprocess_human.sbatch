#!/bin/bash
#SBATCH --job-name=preprocess_human
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/preprocess_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/preprocess_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1

# Create logs directory if it doesn't exist
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

# Change to project directory
cd /users/ssridh26

echo "Starting preprocessing..."
python projects/t2_human_scgraphic/create_pseudobulk_v2.py

echo "Preprocessing complete."
