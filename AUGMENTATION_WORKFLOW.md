# Augmentation Workflow Guide

## Overview

This guide explains **when** and **how** to implement augmentation strategies based on your first fine-tuning results.

## Files Created

1. **`augment_dataset.py`** - Applies post-hoc augmentation to existing NPZ files (downsampling, Poisson noise, diagonal masking)
2. **`create_npz_with_overlap.py`** - Creates NPZ files with overlapping windows (must regenerate from raw data)
3. **`batch_scripts/finetune_augmented.sbatch`** - All-in-one script for augmented fine-tuning

---

## Decision Tree: What to Do After First Fine-tuning

```
Check results in: /users/ssridh26/projects/t2_human_scgraphic/results/GSE238001_*_finetuned/

├─ SCC > 0.6 ──────────> ✅ SUCCESS! Skip augmentation, use model as-is
│
├─ SCC 0.4-0.6 ────────> ⚠️  Implement LIGHT augmentation (Phase 2)
│                         └─> Contact downsampling only
│                         └─> Expected improvement: +0.1 to +0.2 SCC
│
└─ SCC < 0.4 ──────────> ⚠️  Implement AGGRESSIVE augmentation (Phase 3)
                          └─> Contact downsampling + Poisson noise
                          └─> Expected improvement: +0.15 to +0.3 SCC
```

---

## Phase 1: Evaluate First Fine-tuning Results (CURRENT)

### Check Job Status
```bash
# Is job still running?
squeue -u ssridh26

# If completed, check results
ls -lh /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights/checkpoints/
tail -100 /users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_383133.out
```

### Run Inference with Fine-tuned Model
```bash
cd /users/ssridh26/projects/t2_human_scgraphic

# Run on test set (HSC, MPP, LMPP)
python inference.py \
    --dataset GSE238001 \
    --cell-type HSC \
    --checkpoint /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights/checkpoints/best.ckpt \
    --output-dir results/GSE238001_HSC_finetuned

python inference.py \
    --dataset GSE238001 \
    --cell-type MPP \
    --checkpoint /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights/checkpoints/best.ckpt \
    --output-dir results/GSE238001_MPP_finetuned

python inference.py \
    --dataset GSE238001 \
    --cell-type LMPP \
    --checkpoint /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights/checkpoints/best.ckpt \
    --output-dir results/GSE238001_LMPP_finetuned
```

### Compare Results
```bash
# Check full_results.csv for each cell type
cd results

# Pre-trained (baseline)
echo "=== PRE-TRAINED ==="
for ct in HSC MPP LMPP; do
    echo "$ct:"
    grep -E "SCC|SSIM" GSE238001_${ct}/full_results.csv | head -3
done

# Fine-tuned (first round)
echo ""
echo "=== FINE-TUNED ==="
for ct in HSC MPP LMPP; do
    echo "$ct:"
    grep -E "SCC|SSIM" GSE238001_${ct}_finetuned/full_results.csv | head -3
done
```

### Decision Point
Based on average SCC across cell types:
- **SCC > 0.6**: ✅ Stop here, model is good
- **SCC 0.4-0.6**: ⚠️ Proceed to Phase 2
- **SCC < 0.4**: ⚠️ Proceed to Phase 3

---

## Phase 2: Light Augmentation (if SCC 0.4-0.6)

### Strategy
Apply only **contact downsampling** with 2 rates (0.7, 0.5):
- Simulates real experimental variability in sequencing coverage
- 3x data increase: 3,420 → 10,260 samples
- Fast to implement (no regeneration needed)
- High quality (no artificial artifacts)

### Implementation Steps

**Option A: Use batch script (Recommended)**
```bash
cd /users/ssridh26/projects/t2_human_scgraphic

# Edit batch script to use "light" strategy
nano batch_scripts/finetune_augmented.sbatch
# Change: AUGMENTATION_STRATEGY="light"

# Submit job
sbatch batch_scripts/finetune_augmented.sbatch
```

**Option B: Manual steps**
```bash
cd /users/ssridh26/projects/t2_human_scgraphic

# 1. Augment dataset
python augment_dataset.py \
    --input /users/ssridh26/scratch/t2_human_scgraphic/processed/combined_multi_celltype.npz \
    --output /users/ssridh26/scratch/t2_human_scgraphic/processed/combined_augmented_light.npz \
    --augmentations downsample \
    --downsample_rates 0.7 0.5

# 2. Create splits
python create_train_test_val_split.py \
    --input /users/ssridh26/scratch/t2_human_scgraphic/processed/combined_augmented_light.npz \
    --output /users/ssridh26/scratch/t2_human_scgraphic/processed/splits_augmented \
    --negative_threshold 10.0

# 3. Fine-tune
python finetune_human.py \
    --data_dir /users/ssridh26/scratch/t2_human_scgraphic/processed/splits_augmented \
    --checkpoint /oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt \
    --output_dir /users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights_light \
    --experiment human_light_augment \
    --epochs 150 \
    --lr 1e-5
```

### Expected Results
- Training time: 18-36 hours
- Expected SCC: +0.1 to +0.2 improvement
- Target: SCC 0.5-0.7

### What to Do Next
- If SCC > 0.6 after this: ✅ Success, stop here
- If SCC still 0.4-0.6: Consider Phase 3
- If SCC < 0.4: Proceed to Phase 3

---

## Phase 3: Aggressive Augmentation (if SCC < 0.4)

### Strategy
Apply **contact downsampling + Poisson noise**:
- Downsampling: 2 rates (0.7, 0.5)
- Poisson noise: Add sequencing noise (factor=0.1)
- 4x data increase: 3,420 → 13,680 samples
- More diverse training examples
- Risk of introducing artifacts (use carefully)

### Implementation Steps

**Use batch script:**
```bash
cd /users/ssridh26/projects/t2_human_scgraphic

# Edit batch script
nano batch_scripts/finetune_augmented.sbatch
# Change: AUGMENTATION_STRATEGY="aggressive"
# Change: EPOCHS=150  # More epochs for more data

# Submit
sbatch batch_scripts/finetune_augmented.sbatch
```

### Expected Results
- Training time: 24-48 hours
- Expected SCC: +0.15 to +0.3 improvement
- Target: SCC 0.55-0.7

### Additional Options (if still < 0.4)
1. **Add diagonal masking:**
   ```bash
   python augment_dataset.py \
       --input combined_multi_celltype.npz \
       --output combined_augmented_max.npz \
       --augmentations downsample poisson diagonal_mask \
       --downsample_rates 0.7 0.5 \
       --poisson_factor 0.1 \
       --diagonal_mask_distance 10
   ```

2. **Increase training epochs:** 200 with patience=30

3. **Try freezing encoder:**
   ```bash
   python finetune_human.py --freeze_encoder  # Only train decoder
   ```

---

## Phase 4: Window Shifting (Last Resort)

⚠️ **Only if Phases 2-3 don't work and you need even more data**

### Why Last Resort?
- Requires regenerating NPZ files from raw data (slow)
- Creates correlated samples (nearby windows overlap)
- Risk of train/test leakage if not careful
- More complex to implement correctly

### Implementation
```bash
# Regenerate NPZ files with 50% overlap
for ct in HSC MPP LMPP; do
    python create_npz_with_overlap.py \
        --cell_type $ct \
        --window_overlap 0.5 \
        --output_dir /users/ssridh26/scratch/t2_human_scgraphic/processed_overlap50
done

# Combine
python combine_cell_types.py \
    --cell_types HSC MPP LMPP \
    --input_dir /users/ssridh26/scratch/t2_human_scgraphic/processed_overlap50 \
    --output /users/ssridh26/scratch/t2_human_scgraphic/processed/combined_overlap50.npz

# Then apply post-hoc augmentation on top
python augment_dataset.py \
    --input combined_overlap50.npz \
    --output combined_overlap50_augmented.npz \
    --augmentations downsample poisson

# Train
# ... (same as Phase 3)
```

**Expected results:**
- 2x from window overlap × 3x from augmentation = 6x total (20,520 samples)
- Training time: 36-72 hours
- Expected SCC: 0.6-0.8

---

## Quick Reference: Command Cheatsheet

### Check first fine-tuning results
```bash
tail -100 /users/ssridh26/scratch/t2_human_scgraphic/logs/finetune_383133.out | grep -E "SCC|SSIM"
```

### Light augmentation (SCC 0.4-0.6)
```bash
sbatch batch_scripts/finetune_augmented.sbatch  # with AUGMENTATION_STRATEGY="light"
```

### Aggressive augmentation (SCC < 0.4)
```bash
# Edit: AUGMENTATION_STRATEGY="aggressive"
sbatch batch_scripts/finetune_augmented.sbatch
```

### Monitor augmented training
```bash
squeue -u ssridh26
tail -f /users/ssridh26/scratch/t2_human_scgraphic/logs/augment_*.out
```

### Compare multiple fine-tuning rounds
```bash
cd /users/ssridh26/projects/t2_human_scgraphic/results
for dir in GSE238001_HSC*; do
    echo "=== $dir ==="
    grep "Mean SCC" $dir/full_results.csv
done
```

---

## File Locations Reference

```
/users/ssridh26/projects/t2_human_scgraphic/
├── augment_dataset.py              # Post-hoc augmentation script
├── create_npz_with_overlap.py      # Window shifting (regenerate NPZ)
├── batch_scripts/
│   ├── finetune_complete.sbatch     # First fine-tuning (no augmentation)
│   └── finetune_augmented.sbatch    # Augmented fine-tuning (Phase 2/3)
└── results/
    ├── GSE238001_HSC/               # Pre-trained baseline
    ├── GSE238001_HSC_finetuned/     # First fine-tuning
    └── GSE238001_HSC_augmented/     # Augmented fine-tuning

/users/ssridh26/scratch/t2_human_scgraphic/
├── processed/
│   ├── HSC_inference.npz            # Original (1,140 samples)
│   ├── MPP_inference.npz
│   ├── LMPP_inference.npz
│   ├── combined_multi_celltype.npz  # Combined (3,420 samples)
│   ├── combined_augmented_light.npz      # Light (10,260 samples)
│   └── combined_augmented_aggressive.npz # Aggressive (13,680 samples)
├── finetuned_weights/               # First fine-tuning checkpoints
├── finetuned_weights_light/         # Light augmentation checkpoints
└── finetuned_weights_aggressive/    # Aggressive augmentation checkpoints
```

---

## Most Likely Scenario

**Prediction:** Your first fine-tuning will achieve SCC 0.5-0.7, making augmentation unnecessary! 🎉

**Why:**
- 3,420 samples is reasonable for fine-tuning (not from-scratch)
- Transfer learning from mouse weights provides strong initialization
- Quality filtering ensures clean training data
- Early stopping prevents overfitting

**If results are good (SCC > 0.6):**
- Use the fine-tuned model as-is
- No need for any augmentation
- Save time and computational resources
- Focus on downstream analysis and biological interpretation

**Monitor the first training job and make decisions based on actual results!**
