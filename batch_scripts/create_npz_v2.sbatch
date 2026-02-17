#!/bin/bash
#SBATCH --job-name=create_npz_v2
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_npz_v2_%j.log
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_npz_v2_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Create NPZ dataset for human scGrapHiC inference

CELL_TYPE=${1:-HSC}

echo "=========================================="
echo "Creating NPZ Dataset for: $CELL_TYPE"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Activate environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

# Run NPZ creation script
python create_human_npz_v2.py --cell_type $CELL_TYPE

echo ""
echo "End time: $(date)"
echo "=========================================="
