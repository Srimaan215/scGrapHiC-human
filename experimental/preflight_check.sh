#!/bin/bash
# Pre-flight check for fine-tuning

echo "============================================"
echo "FINE-TUNING PRE-FLIGHT CHECK"
echo "============================================"

# Activate environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

echo ""
echo "✓ Checking NPZ files..."
ls -lh /users/ssridh26/scratch/t2_human_scgraphic/processed/*_inference.npz 2>/dev/null || echo "  ⚠️  No NPZ files found!"

echo ""
echo "✓ Checking scripts..."
for script in combine_cell_types.py create_train_test_val_split.py finetune_human.py; do
    if [ -f "$script" ]; then
        echo "  ✅ $script"
    else
        echo "  ❌ $script MISSING!"
    fi
done

echo ""
echo "✓ Checking batch script..."
if [ -f "batch_scripts/finetune_complete.sbatch" ]; then
    echo "  ✅ batch_scripts/finetune_complete.sbatch"
else
    echo "  ❌ batch_scripts/finetune_complete.sbatch MISSING!"
fi

echo ""
echo "✓ Checking checkpoint..."
CKPT="/oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt"
if [ -f "$CKPT" ]; then
    echo "  ✅ Pretrained checkpoint exists"
else
    echo "  ❌ Checkpoint not found: $CKPT"
fi

echo ""
echo "✓ Checking output directories..."
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/processed/splits
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs
echo "  ✅ Output directories ready"

echo ""
echo "============================================"
echo "STATUS SUMMARY"
echo "============================================"

# Count NPZ files
NPZ_COUNT=$(ls /users/ssridh26/scratch/t2_human_scgraphic/processed/*_inference.npz 2>/dev/null | wc -l)
echo "  NPZ files available: $NPZ_COUNT"

if [ $NPZ_COUNT -ge 3 ]; then
    echo ""
    echo "🎉 READY TO START FINE-TUNING!"
    echo ""
    echo "To start, run:"
    echo "  cd /users/ssridh26/projects/t2_human_scgraphic"
    echo "  sbatch batch_scripts/finetune_complete.sbatch"
    echo ""
else
    echo ""
    echo "⚠️  Need at least 3 cell type NPZ files to continue"
    echo "   Currently have: $NPZ_COUNT"
fi
