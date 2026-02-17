#!/bin/bash
#SBATCH --job-name=create_and_infer_2cells
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_infer_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/create_infer_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

# Change to project directory
cd /users/ssridh26/projects/t2_human_scgraphic

echo "============================================"
echo "Creating .npz files and running inference"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

# Create MPP .npz file
echo ""
echo "=================================================="
echo "[1/4] Creating MPP .npz file..."
echo "=================================================="
python create_human_npz_v2.py \
    --cell_type MPP \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/processed

# Create LMPP .npz file  
echo ""
echo "=================================================="
echo "[2/4] Creating LMPP .npz file..."
echo "=================================================="
python create_human_npz_v2.py \
    --cell_type LMPP \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/processed

# Run inference on MPP
echo ""
echo "=================================================="
echo "[3/4] Running inference on MPP..."
echo "=================================================="
python inference.py \
    --dataset GSE238001 \
    --cell-type MPP

# Run inference on LMPP
echo ""
echo "=================================================="
echo "[4/4] Running inference on LMPP..."
echo "=================================================="
python inference.py \
    --dataset GSE238001 \
    --cell-type LMPP

echo ""
echo "============================================"
echo "All processing complete!"
echo "End time: $(date)"
echo "============================================"
