#!/bin/bash
#SBATCH --job-name=augment_finetune
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/augment_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/augment_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# Augmented fine-tuning pipeline
# This implements the strategies from IMPROVEMENTS.md

# Create logs directory
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

echo "============================================"
echo "scGrapHiC Augmented Fine-tuning"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Configuration
CELL_TYPES="HSC MPP LMPP"
AUGMENTATION_STRATEGY="light"  # Options: light, aggressive
EXPERIMENT_NAME="human_augmented_finetune"
EPOCHS=150
LEARNING_RATE=1e-5
BATCH_SIZE=32

# Paths
PROCESSED_DIR="/users/ssridh26/scratch/t2_human_scgraphic/processed"
SPLITS_DIR="${PROCESSED_DIR}/splits_augmented"
WEIGHTS_DIR="/users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights_augmented"

echo ""
echo "Augmentation strategy: ${AUGMENTATION_STRATEGY}"
echo ""

if [ "$AUGMENTATION_STRATEGY" == "light" ]; then
    # Phase 2: Light Augmentation
    # - Contact downsampling: 2 rates (0.7, 0.5)
    # - Expected: 3x original data (3,420 → 10,260 samples)
    
    echo "=================================================="
    echo "Phase 2: Light Augmentation"
    echo "=================================================="
    echo "Applying:"
    echo "  - Contact downsampling (0.7, 0.5)"
    echo "Expected: 3x data (3,420 → 10,260 samples)"
    echo ""
    
    python augment_dataset.py \
        --input ${PROCESSED_DIR}/combined_multi_celltype.npz \
        --output ${PROCESSED_DIR}/combined_augmented_light.npz \
        --augmentations downsample \
        --downsample_rates 0.7 0.5 \
        --seed 42
    
    COMBINED_FILE="${PROCESSED_DIR}/combined_augmented_light.npz"
    
elif [ "$AUGMENTATION_STRATEGY" == "aggressive" ]; then
    # Phase 3: Aggressive Augmentation
    # - Contact downsampling: 2 rates
    # - Poisson noise
    # - Expected: 4x original data (3,420 → 13,680 samples)
    
    echo "=================================================="
    echo "Phase 3: Aggressive Augmentation"
    echo "=================================================="
    echo "Applying:"
    echo "  - Contact downsampling (0.7, 0.5)"
    echo "  - Poisson noise (factor=0.1)"
    echo "Expected: 4x data (3,420 → 13,680 samples)"
    echo ""
    
    python augment_dataset.py \
        --input ${PROCESSED_DIR}/combined_multi_celltype.npz \
        --output ${PROCESSED_DIR}/combined_augmented_aggressive.npz \
        --augmentations downsample poisson \
        --downsample_rates 0.7 0.5 \
        --poisson_factor 0.1 \
        --seed 42
    
    COMBINED_FILE="${PROCESSED_DIR}/combined_augmented_aggressive.npz"
    
else
    echo "ERROR: Unknown augmentation strategy: ${AUGMENTATION_STRATEGY}"
    exit 1
fi

# Create train/val/test splits
echo ""
echo "=================================================="
echo "Creating train/val/test splits..."
echo "=================================================="
python create_train_test_val_split.py \
    --input ${COMBINED_FILE} \
    --output ${SPLITS_DIR} \
    --negative_threshold 10.0

# Fine-tune the model
echo ""
echo "=================================================="
echo "Fine-tuning with augmented data..."
echo "=================================================="
python finetune_human.py \
    --data_dir ${SPLITS_DIR} \
    --checkpoint /oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt \
    --output_dir ${WEIGHTS_DIR} \
    --experiment ${EXPERIMENT_NAME}_${AUGMENTATION_STRATEGY} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --val_every 5 \
    --early_stopping 20 \
    --gradient_clip 1.0

echo ""
echo "============================================"
echo "Augmented fine-tuning complete!"
echo "End time: $(date)"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  Augmented data: ${COMBINED_FILE}"
echo "  Checkpoints: ${WEIGHTS_DIR}/checkpoints"
echo "  Logs: ${WEIGHTS_DIR}/logs"
