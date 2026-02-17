# Fine-tuning scGrapHiC on Human Data - Complete Workflow

## Overview

Fine-tuning adapts the pre-trained mouse model to human data with lower learning rates and human-specific samples. This improves prediction accuracy on human cell types.

## Current Status

✅ **Completed:**
- HSC_inference.npz (1140 samples)
- MPP_inference.npz (1140 samples)
- LMPP_inference.npz (1140 samples)

⏳ **Optional - Create more cell types:**
- MEP_inference.npz
- B_NK_inference.npz

## Fine-tuning Process

### Step 1: Create Additional Cell Type NPZ Files (Optional)

If you want to include MEP and B_NK:

```bash
# Submit job to create MEP and B_NK NPZ files
cd /users/ssridh26/projects/t2_human_scgraphic

# Create MEP
python create_human_npz_v2.py \
    --cell_type MEP \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/processed

# Create B_NK
python create_human_npz_v2.py \
    --cell_type B_NK \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/processed
```

### Step 2: Combine Multiple Cell Types

Combining cell types creates a more diverse training set:

```bash
python combine_cell_types.py \
    --cell_types HSC MPP LMPP \
    --input_dir /users/ssridh26/scratch/t2_human_scgraphic/processed \
    --output /users/ssridh26/scratch/t2_human_scgraphic/processed/combined_multi_celltype.npz
```

**Why combine?**
- Better generalization across cell types
- More training data (3420 samples from 3 cell types)
- Model learns shared hematopoietic features

**Alternative:** Fine-tune on single cell type (e.g., just HSC) for cell-type-specific model

### Step 3: Create Train/Val/Test Splits

Split by chromosome to avoid data leakage:

```bash
python create_train_test_val_split.py \
    --input /users/ssridh26/scratch/t2_human_scgraphic/processed/combined_multi_celltype.npz \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/processed/splits \
    --negative_threshold 10.0 \
    --filter_training_only
```

**Split strategy:**
- **Train:** chr1,2,3,4,5,6,7,8,10,11,14,15,18,20 (~60%)
- **Val:** chr9,12,17,21 (~20%)
- **Test:** chr13,16,19,22 (~20%)

**Quality filtering:**
- Training: Keep only samples with <10% negative values
- Val/Test: Keep all samples (for unbiased evaluation)

### Step 4: Run Fine-tuning

**Option A: All-in-one script (Recommended)**

```bash
sbatch batch_scripts/finetune_complete.sbatch
```

This automatically:
1. Combines cell types
2. Creates splits
3. Runs fine-tuning

**Option B: Manual fine-tuning (if splits already created)**

```bash
python finetune_human.py \
    --data_dir /users/ssridh26/scratch/t2_human_scgraphic/processed/splits \
    --checkpoint /oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights \
    --experiment human_multi_celltype_finetune \
    --epochs 100 \
    --lr 1e-5 \
    --batch_size 32 \
    --val_every 5 \
    --early_stopping 20
```

## Key Parameters

### Learning Rate
- **Mouse pre-training:** 1e-3 (default)
- **Human fine-tuning:** 1e-5 (100x lower)
- Lower LR prevents catastrophic forgetting of mouse features

### Epochs
- **Typical:** 50-100 epochs
- **Early stopping:** Stops if no improvement for 20 epochs
- Monitors validation SCC (Spearman Correlation)

### Batch Size
- **Default:** 32
- **If OOM:** Reduce to 16 or 8
- **If GPU underutilized:** Increase to 64

### Optional: Freeze Encoder

Freeze encoder, only train decoder (faster, less prone to overfitting):

```bash
python finetune_human.py \
    --freeze_encoder \
    ...other args...
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights/logs
```

**Key metrics to watch:**
- `valid/SCC` - Spearman correlation (higher = better)
- `valid/SSIM` - Structural similarity (higher = better)
- `train/loss` - Should decrease steadily
- `learning_rate` - Should stay constant or decay slowly

### Check Job Status

```bash
squeue -u ssridh26
tail -f /users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_*.out
```

## After Fine-tuning

### Step 5: Run Inference with Fine-tuned Model

Use the fine-tuned checkpoint for inference:

```bash
python inference.py \
    --dataset GSE238001 \
    --cell-type HSC \
    --checkpoint /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights/checkpoints/best.ckpt
```

### Step 6: Compare Results

Compare pre-trained vs fine-tuned performance:

```bash
# Results locations:
# Pre-trained: /users/ssridh26/projects/t2_human_scgraphic/results/GSE238001_HSC/
# Fine-tuned: /users/ssridh26/projects/t2_human_scgraphic/results/GSE238001_HSC_finetuned/
```

## Expected Improvements

After fine-tuning on human data:
- **SCC:** 0.3-0.4 → 0.5-0.7 (50-75% improvement)
- **SSIM:** 0.7-0.8 → 0.85-0.95
- **GenomeDISCO:** Improved chromosomal structure similarity
- **TAD boundaries:** Better detection of topologically associating domains

## Troubleshooting

### Issue: OOM (Out of Memory)
**Solution:** Reduce batch size or use gradient accumulation

### Issue: Model not improving
**Solutions:**
- Increase learning rate (try 5e-5)
- Reduce quality filtering threshold
- Add more cell types for diversity

### Issue: Overfitting (train loss << val loss)
**Solutions:**
- Add dropout/regularization
- Freeze encoder weights
- Reduce training data or epochs

### Issue: Taking too long
**Solutions:**
- Use `--freeze_encoder` to train decoder only
- Reduce `--epochs` to 50
- Increase `--batch_size` if GPU allows

## File Structure

```
/users/ssridh26/scratch/t2_human_scgraphic/
├── processed/
│   ├── HSC_inference.npz
│   ├── MPP_inference.npz
│   ├── LMPP_inference.npz
│   ├── combined_multi_celltype.npz  # Created in Step 2
│   └── splits/                       # Created in Step 3
│       ├── HSC_train.npz
│       ├── HSC_val.npz
│       └── HSC_test.npz
├── finetuned_weights/                # Created in Step 4
│   ├── checkpoints/
│   │   ├── best.ckpt
│   │   └── last.ckpt
│   └── logs/
│       └── human_multi_celltype_finetune/
└── logs/
    └── finetune_*.out
```

## Quick Start

**For the impatient - Run everything in one command:**

```bash
cd /users/ssridh26/projects/t2_human_scgraphic
sbatch batch_scripts/finetune_complete.sbatch
```

This will take ~12-24 hours depending on GPU availability and training convergence.
