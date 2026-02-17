#!/bin/bash
#SBATCH --job-name=scgraphic_ft_infer
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/inference_finetuned_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/inference_finetuned_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

# Change to project directory
cd /users/ssridh26/projects/t2_human_scgraphic

# Cell type to process
CELL_TYPE=${CELL_TYPE:-HSC}

# Fine-tuned checkpoint path
CHECKPOINT="/users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights/human_scgraphic.ckpt"

echo "============================================"
echo "scGrapHiC Inference (Fine-tuned on Human)"
echo "Cell type: ${CELL_TYPE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Run inference with fine-tuned weights
# Parameters match the fine-tuning configuration
python inference.py \
    --dataset GSE238001 \
    --cell-type ${CELL_TYPE} \
    --checkpoint ${CHECKPOINT} \
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
