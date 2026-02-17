# Improvements After First Fine-tuning Round

## Decision Tree: When to Augment

```
Fine-tuned model results:
├─ SCC > 0.6 ──────────> ✅ Success! No augmentation needed
├─ SCC 0.4-0.6 ────────> ⚠️  Consider light augmentation
└─ SCC < 0.4 ──────────> ⚠️  Implement augmentation pipeline
```

---

## Hi-C Data Augmentation Strategies

### 1. **Window Shifting (Overlapping Windows)** ⭐ Best first option

**Impact:** 2-3x data increase  
**Quality:** High (uses only real data)  
**Implementation difficulty:** Easy

```python
# Current: Non-overlapping 128-bin windows
# Window at chr1: [0-128], [128-256], [256-384]...

# Augmented: 50% overlap
# Window at chr1: [0-128], [64-192], [128-256], [192-320]...
# Result: 2x more samples

# Implementation in create_human_npz_v2.py
OVERLAP_FRACTION = 0.5  # or 0.25 for 1.33x, 0.75 for 4x
step_size = int(128 * (1 - OVERLAP_FRACTION))

for start_bin in range(0, n_bins - 128, step_size):
    end_bin = start_bin + 128
    # Extract window...
```

**Pros:**
- No data artifacts
- Tests model on shifted contexts
- Simple to implement

**Cons:**
- Correlated samples (adjacent windows overlap)
- Need to ensure validation/test don't overlap with training windows

---

### 2. **Contact Downsampling (Coverage Simulation)**

**Impact:** 1.5-2x data  
**Quality:** High (simulates real experimental variability)  
**Implementation difficulty:** Medium

```python
# Simulate lower sequencing coverage by subsampling contacts
def downsample_contacts(matrix, downsample_rate=0.7):
    """
    Randomly drop 30% of contacts to simulate lower coverage.
    """
    mask = np.random.binomial(1, downsample_rate, matrix.shape)
    downsampled = matrix * mask
    return downsampled

# Apply during data loading
# For each sample, create 1-2 downsampled versions
downsample_rates = [0.7, 0.5]  # 70% and 50% coverage
for rate in downsample_rates:
    augmented_target = downsample_contacts(target_matrix, rate)
    # Keep original features, use downsampled target
    # Model learns to predict from lower coverage
```

**Pros:**
- Matches real experimental noise
- Improves robustness to coverage variation
- Ground truth remains real data

**Cons:**
- Need to be careful not to make data TOO sparse
- May reduce signal in already sparse regions

---

### 3. **Poisson Noise Injection**

**Impact:** 2x data  
**Quality:** Medium-High (biologically realistic)  
**Implementation difficulty:** Medium

```python
# Hi-C contact counts follow Poisson distribution
def add_poisson_noise(matrix, noise_factor=0.1):
    """
    Add Poisson noise to contacts (lambda = value + noise)
    """
    noisy = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] > 0:
                # Lambda = original value with some noise
                lambda_param = max(matrix[i,j] * (1 + np.random.uniform(-noise_factor, noise_factor)), 0.1)
                noisy[i,j] = np.random.poisson(lambda_param)
    return noisy

# Apply to targets
for sample in dataset:
    original = sample['target']
    noisy_version = add_poisson_noise(original, noise_factor=0.1)
    # Add to training set
```

**Pros:**
- Biologically realistic (matches sequencing noise)
- Helps model learn robust features

**Cons:**
- Can introduce artifacts if noise_factor too high
- Slightly changes data distribution

---

### 4. **Symmetric Flip Augmentation**

**Impact:** 2x data  
**Quality:** High (Hi-C is symmetric)  
**Implementation difficulty:** Easy

```python
# Hi-C matrices are symmetric across diagonal
def flip_hic(matrix):
    """Transpose to create mirror image"""
    return matrix.T

# Since Hi-C is symmetric, flipping is information-preserving
# BUT: Most models already learn this symmetry
# Only useful if model architecture doesn't enforce symmetry
```

**Pros:**
- Mathematically valid (Hi-C is symmetric)
- Zero information loss

**Cons:**
- Limited benefit if model already handles symmetry
- scGrapHiC likely already exploits this

---

### 5. **Diagonal Masking / Distance-Based Sampling**

**Impact:** 1.5-2x data  
**Quality:** Medium  
**Implementation difficulty:** Medium

```python
# Train model to handle missing diagonals (common in real data)
def mask_diagonal_bands(matrix, mask_distance=10):
    """
    Randomly mask diagonal bands to simulate missing data.
    """
    masked = matrix.copy()
    # Pick random diagonal to mask (within mask_distance)
    diag_idx = np.random.randint(-mask_distance, mask_distance)
    if diag_idx >= 0:
        masked[np.diag_indices(matrix.shape[0] - diag_idx, ndim=2)] = 0
    else:
        # Lower diagonal
        ...
    return masked

# Creates harder training examples
# Model learns to fill in missing regions
```

**Pros:**
- Simulates real missing data patterns
- Improves imputation capabilities

**Cons:**
- Can make training harder
- May not help if goal is just contact prediction

---

## Implementation Strategy

### Phase 1: Post-First-Fine-tuning Evaluation (You are here)

1. Run fine-tuning with current 3,420 samples
2. Evaluate on test set:
   - **SCC > 0.6:** Success! Stop here
   - **SCC 0.4-0.6:** Proceed to Phase 2
   - **SCC < 0.4:** Proceed to Phase 2 + 3

### Phase 2: Light Augmentation (if SCC < 0.6)

**Quick wins - implement in this order:**

1. **Window shifting** (50% overlap) → 6,840 samples
2. **Contact downsampling** (2 rates) → 10,260 samples
3. Re-run fine-tuning

**Expected improvement:** SCC +0.1 to +0.2

### Phase 3: Aggressive Augmentation (if SCC < 0.4)

1. Add Poisson noise → 20,520 samples
2. More aggressive window shifting (75% overlap) → 40,000+ samples
3. Add diagonal masking
4. **Longer training**: 150-200 epochs with strong regularization

**Expected improvement:** SCC +0.15 to +0.3

---

## Code Implementation Template

### Create `augmented_dataset.py`

```python
import numpy as np
from pathlib import Path

def create_augmented_dataset(
    input_npz,
    output_npz,
    augmentations=['window_shift', 'downsample']
):
    """
    Create augmented version of dataset.
    
    Args:
        input_npz: Path to combined_multi_celltype.npz
        output_npz: Path to save augmented dataset
        augmentations: List of augmentation strategies
    """
    data = np.load(input_npz, allow_pickle=True)
    
    all_features = [data['node_features']]
    all_targets = [data['targets']]
    all_pes = [data['pes']]
    all_bulk = [data['bulk_hics']]
    all_idx = [data['indexes']]
    all_meta = [data['metadatas']]
    
    # Window shifting (if not already done during NPZ creation)
    if 'window_shift' in augmentations:
        # This would require re-processing from pseudobulk
        # See implementation note below
        pass
    
    # Downsampling augmentation
    if 'downsample' in augmentations:
        print("Applying contact downsampling...")
        for rate in [0.7, 0.5]:
            downsampled_targets = []
            for target in data['targets']:
                mask = np.random.binomial(1, rate, target.shape)
                downsampled = target * mask
                downsampled_targets.append(downsampled)
            
            all_features.append(data['node_features'])
            all_targets.append(np.array(downsampled_targets))
            all_pes.append(data['pes'])
            all_bulk.append(data['bulk_hics'])
            all_idx.append(data['indexes'])
            all_meta.append(data['metadatas'])
    
    # Poisson noise
    if 'poisson' in augmentations:
        print("Applying Poisson noise...")
        noisy_targets = []
        for target in data['targets']:
            noisy = apply_poisson_noise(target, 0.1)
            noisy_targets.append(noisy)
        
        all_features.append(data['node_features'])
        all_targets.append(np.array(noisy_targets))
        # ... append others
    
    # Combine all augmented versions
    combined = {
        'node_features': np.concatenate(all_features, axis=0),
        'targets': np.concatenate(all_targets, axis=0),
        'pes': np.concatenate(all_pes, axis=0),
        'bulk_hics': np.concatenate(all_bulk, axis=0),
        'indexes': np.concatenate(all_idx, axis=0),
        'metadatas': np.concatenate(all_meta, axis=0)
    }
    
    np.savez(output_npz, **combined)
    print(f"Augmented dataset: {combined['targets'].shape[0]} samples")
```

### Modify `create_human_npz_v2.py` for Window Shifting

```python
# In create_human_npz_v2.py, modify the windowing loop:

WINDOW_SIZE = 128
OVERLAP = 64  # 50% overlap (or 32 for 25%, 96 for 75%)

for chrom in chromosomes:
    n_bins = chrom_size // resolution
    
    # Original: start_bin in range(0, n_bins - WINDOW_SIZE, WINDOW_SIZE)
    # Augmented:
    for start_bin in range(0, n_bins - WINDOW_SIZE, WINDOW_SIZE - OVERLAP):
        end_bin = start_bin + WINDOW_SIZE
        
        # Extract window...
        # IMPORTANT: Track window positions to prevent train/test leakage
        # Solution: Split on chromosome first, then apply windowing
```

---

## Critical Notes

### Avoiding Data Leakage with Overlapping Windows

**Problem:** Overlapping windows in train AND test = information leakage

**Solution:** Apply chromosome splits BEFORE windowing
```python
# Correct order:
1. Split chromosomes: train_chrs, val_chrs, test_chrs
2. For each split, create overlapping windows
3. This ensures no window overlaps between train and test

# In create_train_test_val_split.py:
# Already splits by chromosome ✓
# Just need to ensure windowing happens AFTER split
```

### When NOT to Augment

**Don't augment if:**
- First fine-tuning gives SCC > 0.6
- Model is clearly overfitting (train SCC >> test SCC)
- You're planning to collect more real data

**Augmentation is a last resort when:**
- Real data collection impossible (your case)
- Model underfitting (train and test SCC both low)
- Need more diversity in training examples

---

## Expected Results

### Conservative Estimate (assumes SCC after first fine-tuning = 0.4)

| Strategy | Samples | Expected SCC | Training Time |
|----------|---------|--------------|---------------|
| Baseline (current) | 3,420 | 0.4-0.6 | 12-24h |
| + Window shift (50%) | 6,840 | 0.5-0.7 | 18-36h |
| + Downsampling (2x) | 13,680 | 0.55-0.75 | 24-48h |
| + Poisson noise | 27,360 | 0.6-0.8 | 36-72h |
| + Aggressive (75% overlap) | 40,000+ | 0.65-0.85 | 48-96h |

### Optimistic Estimate (assumes first fine-tuning works well)

- First fine-tuning: SCC 0.6-0.7 ✓ **No augmentation needed!**

---

## Next Steps After Fine-tuning Results

```bash
# After fine-tuning completes:

# 1. Check results
cd /users/ssridh26/projects/t2_human_scgraphic/results
# Look for GSE238001_{HSC,MPP,LMPP}_finetuned folders

# 2. Compare SCC scores
python analyze_results.py --compare pretrained finetuned

# 3. Decision:
# - SCC > 0.6: Done! ✓
# - SCC 0.4-0.6: Implement window shifting
# - SCC < 0.4: Implement full augmentation pipeline
```

---

## Quick Reference: Augmentation Checklist

**If you need to augment, do in this order:**

- [ ] **Step 1:** Window shifting (easiest, biggest impact)
  - Modify `create_human_npz_v2.py` 
  - Set `OVERLAP = 64` (50% overlap)
  - Regenerate NPZ files

- [ ] **Step 2:** Contact downsampling (if still needed)
  - Create `augmented_dataset.py`
  - Downsample rates: [0.7, 0.5]
  - Apply to combined dataset

- [ ] **Step 3:** Poisson noise (if still needed)
  - Add to `augmented_dataset.py`
  - noise_factor = 0.1
  - Test on small subset first

- [ ] **Step 4:** Re-run fine-tuning
  - Use augmented dataset
  - May need longer training (150-200 epochs)
  - Monitor for overfitting (early stopping still on)

**Most likely scenario:** First fine-tuning works well enough, augmentation not needed! 🎉
