#!/bin/bash
#SBATCH -J human_inference_test
#SBATCH -o /users/ssridh26/jobtmp/human_inference_test_%j.out
#SBATCH -e /users/ssridh26/jobtmp/human_inference_test_%j.err
#SBATCH -t 01:00:00
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1

echo "=============================================="
echo "Human scGrapHiC Zero-shot Inference (t2)"
echo "=============================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

source ~/miniforge3/bin/activate scgraphic_env
export PYTHONUNBUFFERED=1

cd /users/ssridh26/projects/t2_human_scgraphic

# Run inference using the proper t2_human_scgraphic inference.py
python inference.py \
  --npz-file /users/ssridh26/scratch/t2_human_scgraphic/processed/HSC_inference.npz \
  --cell-type HSC \
  --experiment human_blood_zero_shot \
  --rna_seq True \
  --positional_encodings True \
  --ctcf_motif True \
  --cpg_motif True

echo ""
echo "=============================================="
echo "Inference Complete"
echo "End time: $(date)"
