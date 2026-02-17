#!/bin/bash
#SBATCH --job-name=full_inf_test
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/full_inference_test_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/full_inference_test_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Quick test: Run inference on ALL chromosomes with existing K562 bulk files
# to see if full chromosome coverage improves SCC from 25.30% (chr13,16,19 only)

echo "============================================"
echo "Full Chromosome Inference Test (K562 baseline)"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "============================================"

source ~/miniforge3/bin/activate
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

CHECKPOINT="/users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights_gm12878/checkpoints/epoch=99-valid/SCC=0.2263.ckpt"

echo ""
echo "Running inference on ALL chromosomes (chr1-20)..."
echo "Using existing K562 bulk files for quick baseline"
echo "Checkpoint: ${CHECKPOINT}"
echo ""

python run_full_inference.py \
    --checkpoint ${CHECKPOINT} \
    --cell_types HSC MPP LMPP \
    --batch_size 32 \
    --experiment_name human_gm12878_full_k562 &

# Store the PID to monitor the process
PYTHON_PID=$!

echo ""
echo "Monitoring job for first 5 minutes to catch early failures..."
echo "PID: ${PYTHON_PID}"
echo ""

# Monitor for 5 minutes
for i in {1..10}; do
    sleep 30
    if ! kill -0 ${PYTHON_PID} 2>/dev/null; then
        echo "⚠️  Process ended early at $(date)"
        wait ${PYTHON_PID}
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "❌ ERROR: Process failed with exit code ${EXIT_CODE}"
            echo "Check error log for details"
            exit $EXIT_CODE
        else
            echo "✓ Process completed successfully (quick exit)"
            break
        fi
    fi
    echo "  [$(date)] Still running... (${i}/10 checks)"
done

if kill -0 ${PYTHON_PID} 2>/dev/null; then
    echo "✓ Process survived first 5 minutes, continuing in background..."
    wait ${PYTHON_PID}
fi

echo ""
echo "============================================"
echo "Complete: $(date)"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  /users/ssridh26/projects/t2_human_scgraphic/results/human_gm12878_full_k562_*/"
echo ""
echo "Next steps:"
echo "1. Check average SCC across all chromosomes"
echo "2. Compare with test split results (25.30%, chr13/16/19 only)"
echo "3. If improved, regenerate with GM12878 bulk for final results"
