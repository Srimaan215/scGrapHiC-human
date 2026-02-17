#!/bin/bash
#SBATCH --job-name=full_inference_gm12878
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/full_inference_gm12878_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/full_inference_gm12878_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Full inference across ALL chromosomes (chr1-chr22) for all cell types
# This gives true performance assessment vs limited test split (chr13,16,19)

echo "============================================"
echo "scGrapHiC Full Inference - GM12878 Bulk"
echo "Submit time: $(date)"
echo "Node: $(hostname)"
echo "============================================"

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

# Configuration
CHECKPOINT="/users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights_gm12878/checkpoints/epoch=99-valid/SCC=0.2263.ckpt"
CELL_TYPES="HSC MPP LMPP"
BATCH_SIZE=32

echo ""
echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Cell types: ${CELL_TYPES}"
echo "  Batch size: ${BATCH_SIZE}"
echo ""

# Check if GM12878 inference files exist
NEED_REGENERATION=false
for cell_type in ${CELL_TYPES}; do
    if [ ! -f "/users/ssridh26/scratch/t2_human_scgraphic/processed/${cell_type}_inference_gm12878.npz" ]; then
        echo "⚠️  ${cell_type}_inference_gm12878.npz not found"
        NEED_REGENERATION=true
    fi
done

if [ "$NEED_REGENERATION" = true ]; then
    echo ""
    echo "============================================"
    echo "STEP 1: Regenerating NPZ files with GM12878 bulk"
    echo "============================================"
    echo ""
    
    for cell_type in ${CELL_TYPES}; do
        echo "Generating ${cell_type}_inference_gm12878.npz..."
        python create_human_npz_v2.py \
            --cell_type ${cell_type} \
            --output_suffix _gm12878 \
            --include_all_chromosomes
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to generate ${cell_type}"
            exit 1
        fi
        echo "✓ ${cell_type} completed"
        echo ""
    done
fi

echo ""
echo "============================================"
echo "STEP 2: Running Full Inference"
echo "============================================"
echo ""

python run_full_inference.py \
    --checkpoint ${CHECKPOINT} \
    --cell_types ${CELL_TYPES} \
    --use_gm12878 \
    --batch_size ${BATCH_SIZE} \
    --experiment_name human_gm12878_full_inference

echo ""
echo "============================================"
echo "Inference complete!"
echo "End time: $(date)"
echo "============================================"
echo ""
echo "View results in:"
echo "  /users/ssridh26/projects/t2_human_scgraphic/results/human_gm12878_full_inference_*/"
echo ""
echo "Compare with limited test split (chr13,16,19 only):"
echo "  Previous (test split): /users/ssridh26/projects/t2_human_scgraphic/results/human_gm12878_bulk_finetune/"
