#!/bin/bash
#SBATCH --job-name=finetune_human
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# Create logs directory
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

echo "============================================"
echo "scGrapHiC Fine-tuning on Human Data"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Configuration
CELL_TYPES="HSC MPP LMPP"  # Space-separated list
EXPERIMENT_NAME="human_multi_celltype_finetune"
EPOCHS=100
LEARNING_RATE=1e-5
BATCH_SIZE=32

echo ""
echo "Configuration:"
echo "  Cell types: ${CELL_TYPES}"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size: ${BATCH_SIZE}"

# Step 1: Combine cell types (if not already done)
echo ""
echo "=================================================="
echo "[1/3] Combining cell type datasets..."
echo "=================================================="
python combine_cell_types.py \
    --cell_types ${CELL_TYPES} \
    --input_dir /users/ssridh26/scratch/t2_human_scgraphic/processed \
    --output /users/ssridh26/scratch/t2_human_scgraphic/processed/combined_multi_celltype.npz

# Step 2: Create train/val/test splits
echo ""
echo "=================================================="
echo "[2/3] Creating train/val/test splits..."
echo "=================================================="
python create_train_test_val_split.py \
    --input /users/ssridh26/scratch/t2_human_scgraphic/processed/combined_multi_celltype.npz \
    --output /users/ssridh26/scratch/t2_human_scgraphic/processed/splits \
    --negative_threshold 10.0

# Step 3: Fine-tune the model
echo ""
echo "=================================================="
echo "[3/3] Fine-tuning model..."
echo "=================================================="
python finetune_human.py \
    --data_dir /users/ssridh26/scratch/t2_human_scgraphic/processed/splits \
    --checkpoint /oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights \
    --experiment ${EXPERIMENT_NAME} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --val_every 5 \
    --early_stopping 20 \
    --gradient_clip 1.0

echo ""
echo "============================================"
echo "Fine-tuning complete!"
echo "End time: $(date)"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  Checkpoints: /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights/checkpoints"
echo "  Logs: /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights/logs"
