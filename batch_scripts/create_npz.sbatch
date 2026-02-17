#!/bin/bash
#SBATCH --job-name=create_npz
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_npz_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_npz_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --dependency=afterok:334950

# Create logs directory if it doesn't exist
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

# Change to project directory
cd /users/ssridh26

echo "Starting NPZ creation..."
python projects/t2_human_scgraphic/create_human_npz_v3.py

echo "NPZ creation complete."
