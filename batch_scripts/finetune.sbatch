#!/bin/bash
#SBATCH --job-name=scgraphic_finetune
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_%j.log
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load conda environment
source /users/ssridh26/miniforge3/bin/activate scgraphic_env

# Change to project directory
cd /users/ssridh26/projects/t2_human_scgraphic

# Create logs directory if it doesn't exist
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs

# Run fine-tuning
echo "Starting fine-tuning at $(date)"
echo "Job ID: $SLURM_JOB_ID"

python finetune_human.py \
    --experiment human_hsc_finetune_v1 \
    --epochs 200 \
    --lr 1e-5 \
    --batch_size 32 \
    --val_every 10 \
    --early_stopping 30 \
    --checkpoint /oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt \
    --data_dir /users/ssridh26/scratch/t2_human_scgraphic/processed/splits \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights

echo "Fine-tuning completed at $(date)"
