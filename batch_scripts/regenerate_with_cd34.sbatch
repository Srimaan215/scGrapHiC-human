#!/bin/bash
#SBATCH --job-name=regen_cd34
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/regenerate_cd34_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/regenerate_cd34_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

# Regenerate NPZ files with CD34+ bulk Hi-C
# Then combine, split, and prepare for fine-tuning

# Create logs directory
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

echo "============================================"
echo "Regenerating NPZ Files with GM12878 Bulk Hi-C"
echo "Start time: $(date)"
echo "============================================"

# Paths
PROCESSED_DIR="/users/ssridh26/scratch/t2_human_scgraphic/processed"
SPLITS_DIR="${PROCESSED_DIR}/splits_gm12878"

# Step 1: Regenerate each cell type with CD34+ bulk
echo ""
echo "Step 1: Regenerating HSC with CD34+ bulk..."
python create_human_npz_v2.py --cell_type HSC
if [ $? -ne 0 ]; then
    echo "❌ HSC regeneration failed"
    exit 1
fi
echo "✓ HSC completed"

echo ""
echo "Step 2: Regenerating MPP with CD34+ bulk..."
python create_human_npz_v2.py --cell_type MPP
if [ $? -ne 0 ]; then
    echo "❌ MPP regeneration failed"
    exit 1
fi
echo "✓ MPP completed"

echo ""
echo "Step 3: Regenerating LMPP with CD34+ bulk..."
python create_human_npz_v2.py --cell_type LMPP
if [ $? -ne 0 ]; then
    echo "❌ LMPP regeneration failed"
    exit 1
fi
echo "✓ LMPP completed"

# Step 4: Check bulk mismatch with CD34+
echo ""
echo "Step 4: Checking bulk mismatch with GM12878..."
python check_bulk_mismatch.py > ${PROCESSED_DIR}/gm12878_mismatch_report.txt
echo "✓ Mismatch report saved to: ${PROCESSED_DIR}/gm12878_mismatch_report.txt"
echo ""
echo "Top mismatched samples:"
head -30 ${PROCESSED_DIR}/gm12878_mismatch_report.txt

# Step 5: Combine cell types
echo ""
echo "Step 5: Combining cell types..."
python combine_cell_types.py \
    --cell_types HSC MPP LMPP \
    --input_dir ${PROCESSED_DIR} \
    --output ${PROCESSED_DIR}/combined_gm12878.npz

if [ $? -ne 0 ]; then
    echo "❌ Combining failed"
    exit 1
fi
echo "✓ Combined dataset saved"

# Step 6: Create train/val/test splits
echo ""
echo "Step 6: Creating train/val/test splits..."
mkdir -p ${SPLITS_DIR}

python create_train_test_val_split.py \
    --input ${PROCESSED_DIR}/combined_gm12878.npz \
    --output ${SPLITS_DIR}/combined

if [ $? -ne 0 ]; then
    echo "❌ Split creation failed"
    exit 1
fi
echo "✓ Splits created"

echo ""
echo "============================================"
echo "NPZ Regeneration Complete!"
echo "End time: $(date)"
echo "============================================"

echo ""
echo "Files created:"
ls -lh ${PROCESSED_DIR}/*_inference.npz
echo ""
echo "Combined dataset:"
ls -lh ${PROCESSED_DIR}/combined_cd34.npz
echo ""
echo "Splits:"
ls -lh ${SPLITS_DIR}/

echo ""
echo "Mismatch report summary:"
echo "========================"
grep -E "samples mismatched|Top|chr" ${PROCESSED_DIR}/gm12878_mismatch_report.txt | head -15

echo ""
echo "Next: Submit fine-tuning job"
echo "  sbatch batch_scripts/finetune_gm12878.sbatch"
