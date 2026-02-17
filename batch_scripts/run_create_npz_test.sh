#!/bin/bash
#SBATCH -J create_npz_test
#SBATCH -o /users/ssridh26/jobtmp/create_npz_test_%j.out
#SBATCH -e /users/ssridh26/jobtmp/create_npz_test_%j.err
#SBATCH -t 02:00:00
#SBATCH --mem=64G
#SBATCH -n 4

echo "=============================================="
echo "Create Human NPZ - Test (HSC only)"
echo "=============================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

source ~/miniforge3/bin/activate scgraphic_env
export PYTHONUNBUFFERED=1

cd /users/ssridh26/projects/t2_human_scgraphic

# Create output directory
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/processed

# Run for HSC cell type
python create_human_npz_v2.py --cell_type HSC --output_dir /users/ssridh26/scratch/t2_human_scgraphic/processed

echo ""
echo "=============================================="
echo "Complete"
echo "End time: $(date)"
