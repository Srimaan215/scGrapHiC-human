#!/bin/bash
#SBATCH --job-name=scgraphic_infer
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/inference_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/inference_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

# Change to project directory
cd /users/ssridh26/projects/t2_human_scgraphic

# Cell type to process (default: HSC, can be overridden with --export=CELL_TYPE=MPP)
CELL_TYPE=${CELL_TYPE:-HSC}

echo "============================================"
echo "scGrapHiC Inference"
echo "Cell type: ${CELL_TYPE}"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Run inference with parameters matching the checkpoint
# Key parameters from checkpoint (fixed-cpg-processing):
#   - rna_seq: True
#   - positional_encodings: True
#   - ctcf_motif: True
#   - cpg_motif: True
#   - use_bulk: True
#   - normalization_algorithm: library_size_normalization
#   - num_nodes: 128
#   - pos_encodings_dim: 16
#   - node_features: 5

python inference.py \
    --dataset GSE238001 \
    --cell-type ${CELL_TYPE} \
    --rna_seq True \
    --positional_encodings True \
    --ctcf_motif True \
    --cpg_motif True \
    --use_bulk True \
    --normalization_algorithm library_size_normalization \
    --num_nodes 128 \
    --pos_encodings_dim 16 \
    --node_features 2 \
    --batch_size 64

echo "============================================"
echo "End time: $(date)"
echo "============================================"
