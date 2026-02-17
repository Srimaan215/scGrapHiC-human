#!/bin/bash
#SBATCH --job-name=gen_inf_gm12878
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/gen_inference_gm12878_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/gen_inference_gm12878_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Generate GM12878 NPZ files for all cell types, then run inference

echo "============================================"
echo "Generate GM12878 Inference Files & Run Inference"
echo "Start: $(date)"
echo "============================================"

source ~/miniforge3/bin/activate
conda activate scgraphic_env
cd /users/ssridh26/projects/t2_human_scgraphic

mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Generate NPZ files with GM12878 bulk
echo ""
echo "Step 1: Generating NPZ files with GM12878 bulk..."
for CELL_TYPE in HSC MPP LMPP; do
    echo "  Processing ${CELL_TYPE}..."
    python create_human_npz_v2.py \
        --cell_type ${CELL_TYPE} \
        --output_dir /users/ssridh26/scratch/t2_human_scgraphic/processed
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate ${CELL_TYPE}"
        exit 1
    fi
    
    # Rename to distinguish from old K562 files
    mv /users/ssridh26/scratch/t2_human_scgraphic/processed/${CELL_TYPE}_inference.npz \
       /users/ssridh26/scratch/t2_human_scgraphic/processed/${CELL_TYPE}_inference_gm12878.npz
    
    echo "  ✓ ${CELL_TYPE}_inference_gm12878.npz created"
done

echo ""
echo "✓ All NPZ files generated"
echo ""
echo "Step 2: Running inference on all chromosomes..."

CHECKPOINT="/users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights_gm12878/checkpoints/epoch=99-valid/SCC=0.2263.ckpt"

for CELL_TYPE in HSC MPP LMPP; do
    echo ""
    echo "Running inference: ${CELL_TYPE}"
    python inference.py \
        --dataset GSE238001 \
        --cell-type ${CELL_TYPE} \
        --npz-file /users/ssridh26/scratch/t2_human_scgraphic/processed/${CELL_TYPE}_inference_gm12878.npz \
        --checkpoint ${CHECKPOINT}
    
    if [ $? -ne 0 ]; then
        echo "❌ Inference failed for ${CELL_TYPE}"
        exit 1
    fi
    
    # Calculate average SCC
    RESULTS_FILE="/users/ssridh26/projects/t2_human_scgraphic/results/GSE238001_${CELL_TYPE}/results.csv"
    if [ -f "${RESULTS_FILE}" ]; then
        AVG_SCC=$(awk -F',' 'NR>1 {sum+=$NF; count++} END {print sum/count}' ${RESULTS_FILE})
        echo "  → Average SCC: ${AVG_SCC}"
    fi
done

echo ""
echo "============================================"
echo "Complete: $(date)"
echo "============================================"
echo ""
echo "Results:"
for CELL_TYPE in HSC MPP LMPP; do
    RESULTS_FILE="/users/ssridh26/projects/t2_human_scgraphic/results/GSE238001_${CELL_TYPE}/results.csv"
    if [ -f "${RESULTS_FILE}" ]; then
        AVG=$(awk -F',' 'NR>1 {sum+=$NF; count++} END {printf "%.4f (%.2f%%)", sum/count, sum/count*100}' ${RESULTS_FILE})
        echo "  ${CELL_TYPE}: ${AVG}"
    fi
done
