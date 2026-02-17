#!/bin/bash
#SBATCH --job-name=finetune_cd34
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_cd34_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_cd34_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# Fine-tuning with CD34+ bulk Hi-C (proper biological match for HSC/MPP/LMPP)

# Create logs directory
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

echo "============================================"
echo "scGrapHiC Fine-tuning with CD34+ Bulk Hi-C"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Configuration
CELL_TYPES="HSC MPP LMPP"
EXPERIMENT_NAME="human_cd34_bulk_finetune"
EPOCHS=100
LEARNING_RATE=1e-5
BATCH_SIZE=32

# Paths
PROCESSED_DIR="/users/ssridh26/scratch/t2_human_scgraphic/processed"
SPLITS_DIR="${PROCESSED_DIR}/splits_cd34/combined"
WEIGHTS_DIR="/users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights_cd34"

# Check that splits exist
if [ ! -d "${SPLITS_DIR}" ]; then
    echo "❌ ERROR: Splits directory not found: ${SPLITS_DIR}"
    echo "Run regenerate_with_cd34.sbatch first!"
    exit 1
fi

# Create weights directory
mkdir -p ${WEIGHTS_DIR}

echo ""
echo "Configuration:"
echo "  Cell types: ${CELL_TYPES}"
echo "  Bulk Hi-C: CD34+ (hematopoietic progenitors)"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Batch size: ${BATCH_SIZE}"
echo ""
echo "Paths:"
echo "  Splits: ${SPLITS_DIR}"
echo "  Output: ${WEIGHTS_DIR}"
echo ""

# Run fine-tuning
echo "Starting fine-tuning..."
python finetune_human.py \
    --checkpoint /oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt \
    --data_dir ${SPLITS_DIR} \
    --output_dir ${WEIGHTS_DIR} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --val_every 5 \
    --early_stopping 20 \
    --experiment ${EXPERIMENT_NAME}

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "FINE-TUNING COMPLETE"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "FINE-TUNING FAILED"
    echo "============================================================"
    exit 1
fi

echo ""
echo "============================================"
echo "Fine-tuning complete!"
echo "End time: $(date)"
echo "============================================"

echo ""
echo "Results saved to:"
echo "  Checkpoints: ${WEIGHTS_DIR}/checkpoints"
echo "  Logs: ${WEIGHTS_DIR}/logs"
echo ""
echo "Best model checkpoint:"
ls -lh ${WEIGHTS_DIR}/checkpoints/epoch=*-valid*.ckpt 2>/dev/null | head -3

echo ""
echo "Next: Run inference and compare with K562 baseline"
echo "  python inference.py --checkpoint ${WEIGHTS_DIR}/checkpoints/epoch=XX-valid_SCC=0.XXXX.ckpt"
