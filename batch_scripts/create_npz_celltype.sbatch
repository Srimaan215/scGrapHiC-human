#!/bin/bash
#SBATCH --job-name=create_npz_celltype
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_npz_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_npz_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1

# Create logs directory if it doesn't exist
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

# Change to project directory
cd /users/ssridh26/projects/t2_human_scgraphic

# Cell type to process
CELL_TYPE=${CELL_TYPE:-MPP}

echo "============================================"
echo "Creating .npz file for cell type: ${CELL_TYPE}"
echo "Start time: $(date)"
echo "============================================"

python create_human_npz_v2.py \
    --cell_type ${CELL_TYPE} \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/processed

echo "============================================"
echo "End time: $(date)"
echo "============================================"
