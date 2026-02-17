#!/bin/bash
#SBATCH -J human_scgraphic_full
#SBATCH -o /users/ssridh26/jobtmp/human_full_%j.out
#SBATCH -e /users/ssridh26/jobtmp/human_full_%j.err
#SBATCH -t 02:00:00
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1

echo "=============================================="
echo "Human scGrapHiC Full Pipeline"
echo "=============================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"

# Activate environment
source /users/ssridh26/miniforge3/etc/profile.d/conda.sh
conda activate scgraphic_env

cd /users/ssridh26/projects/t2_human_scgraphic

# ===========================================
# Step 1: Create NPZ dataset with normalized features
# ===========================================
echo ""
echo "Step 1: Creating NPZ dataset..."
echo "=============================================="

python create_human_npz_v2.py --cell_type HSC

# Check if NPZ was created
NPZ_FILE="/users/ssridh26/scratch/t2_human_scgraphic/processed/HSC_inference.npz"
if [ ! -f "$NPZ_FILE" ]; then
    echo "ERROR: NPZ file not created!"
    exit 1
fi

echo ""
echo "NPZ created successfully: $NPZ_FILE"

# Quick verification of features
python3 -c "
import numpy as np
data = np.load('$NPZ_FILE')
nf = data['node_features']
print('Node features shape:', nf.shape)
for i in range(nf.shape[2]):
    vals = nf[..., i]
    print(f'  Ch {i}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}')
print('\\nTargets: min={:.4f}, max={:.4f}'.format(data['targets'].min(), data['targets'].max()))
"

# ===========================================
# Step 2: Run inference
# ===========================================
echo ""
echo "Step 2: Running inference..."
echo "=============================================="

python inference.py \
    --dataset GSE238001 \
    --cell-type HSC \
    --npz-file "$NPZ_FILE" \
    --checkpoint /oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "End time: $(date)"
echo "=============================================="
