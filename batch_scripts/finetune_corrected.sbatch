#!/bin/bash
#SBATCH --job-name=finetune_corrected
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_corrected_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_corrected_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# Fine-tuning with corrected bulk Hi-C (masked in sparse regions)
# This fixes the hallucination problem where model predicts contacts in empty regions

# Create logs directory
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

echo "============================================"
echo "scGrapHiC Fine-tuning with Corrected Bulk Hi-C"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Configuration
CELL_TYPES="HSC MPP LMPP"
EXPERIMENT_NAME="human_corrected_bulk_finetune"
EPOCHS=100
LEARNING_RATE=1e-5
BATCH_SIZE=32

# Paths
PROCESSED_DIR="/users/ssridh26/scratch/t2_human_scgraphic/processed"
SPLITS_DIR="${PROCESSED_DIR}/splits_corrected"
WEIGHTS_DIR="/users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights_corrected"

echo ""
echo "Step 0: Correcting bulk Hi-C in sparse regions..."
echo "=================================================="
python create_corrected_npz.py

echo ""
echo "Step 1: Combining corrected cell types..."
echo "=================================================="
python combine_cell_types.py \
    --cell_types ${CELL_TYPES} \
    --input_dir ${PROCESSED_DIR} \
    --output ${PROCESSED_DIR}/combined_corrected.npz \
    --suffix _corrected

echo ""
echo "Step 2: Creating train/val/test splits..."
echo "=================================================="
python create_train_test_val_split.py \
    --input ${PROCESSED_DIR}/combined_corrected.npz \
    --output ${SPLITS_DIR} \
    --negative_threshold 10.0

echo ""
echo "Step 3: Fine-tuning with corrected data..."
echo "=================================================="
python finetune_human.py \
    --data_dir ${SPLITS_DIR} \
    --checkpoint /oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt \
    --output_dir ${WEIGHTS_DIR} \
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
echo "  Corrected NPZ: ${PROCESSED_DIR}/combined_corrected.npz"
echo "  Checkpoints: ${WEIGHTS_DIR}/checkpoints"
echo "  Logs: ${WEIGHTS_DIR}/logs"
echo ""
echo "Next: Run inference with corrected model and compare:"
echo "  python inference.py --checkpoint ${WEIGHTS_DIR}/checkpoints/last.ckpt"
