# scGrapHiC Human Fine-tuning Repository Structure

## Core Working Files

### Main Scripts
- **`finetune_human.py`** - Main fine-tuning script for human CD34+ cells with bulk Hi-C
- **`inference.py`** - Inference script for running predictions on new data
- **`create_human_npz_v2.py`** - NPZ dataset generation with configurable bulk Hi-C sources
- **`create_train_test_val_split.py`** - Chromosome-stratified train/validation/test splitting
- **`combine_cell_types.py`** - Combine multiple cell type NPZ files
- **`compare_distributions.py`** - Analyze chromosome and cell type distributions

### Batch Scripts (`batch_scripts/`)
Working SLURM batch submission scripts:
- **`finetune_gm12878.sbatch`** - Fine-tune with GM12878 bulk Hi-C (WORKING ✓)
- **`simple_full_inference.sbatch`** - Generate NPZ + run inference (WORKING ✓)
- Other batch scripts for various workflows

### Source Code (`src/`)
Core model implementation:
- `model.py` - scGrapHiC PyTorch Lightning model
- `evaluations.py` - Evaluation metrics (SCC, MSE, SSIM, GD)
- `visualizations.py` - Result logging and visualization
- `globals.py` - Configuration and paths
- `graph_pe.py` - Graph positional encodings
- `TADs.py` - TAD detection utilities

### Documentation
- **`FINETUNING_GUIDE.md`** - Complete fine-tuning workflow guide
- **`FULL_INFERENCE_PLAN.md`** - Full chromosome inference strategy
- **`AUGMENTATION_WORKFLOW.md`** - Dataset augmentation approaches
- **`IMPROVEMENTS.md`** - Model improvement ideas

## Experimental / Archive (`experimental/`)

Scripts that were tested but not actively used:
- Alternative NPZ generation scripts (v1, v3, v4)
- Pseudobulk creation variants
- Analysis utilities (sparsity, bulk mismatch checking)
- Feature inspection tools
- Download and preprocessing scripts
- Alternative inference implementations

These are kept for reference but not required for the main workflow.

## Current Model Performance

**Training Configuration:**
- Dataset: CD34+ hematopoietic cells from GSE238001 (HSC, MPP, LMPP)
- Total cells: 2,815
- Bulk Hi-C: GM12878 (B-lymphocyte)
- Resolution: 50kb bins
- Library size: 25,000 contacts

**Results (Job 396506 - 100 epochs):**
- Validation SCC: 22.63% (chr9, 12, 17)
- Test SCC: 25.30% (chr13, 16, 19)
- Full genome SCC: ~17% (chr1-20)

**Key Finding:**
The chromosome-based data split shows significant variation in performance across chromosomes. The 17% full genome average is more representative than the 22.63% validation SCC which only covered 3 specific chromosomes.

## Quick Start

### 1. Generate Training Data
```bash
# Generate NPZ files for each cell type with GM12878 bulk
python create_human_npz_v2.py --cell_type HSC --output_dir /path/to/output
python create_human_npz_v2.py --cell_type MPP --output_dir /path/to/output
python create_human_npz_v2.py --cell_type LMPP --output_dir /path/to/output

# Combine cell types
python combine_cell_types.py --output /path/to/combined.npz

# Create train/val/test splits
python create_train_test_val_split.py --input combined.npz --output_dir /path/to/splits/
```

### 2. Fine-tune Model
```bash
sbatch batch_scripts/finetune_gm12878.sbatch
```

### 3. Run Inference
```bash
sbatch batch_scripts/simple_full_inference.sbatch
```

## Data Location (Oscar HPC)
- **Processed data**: `/users/ssridh26/scratch/t2_human_scgraphic/processed/`
- **Model weights**: `/users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights_gm12878/`
- **Results**: `/users/ssridh26/scratch/t2_human_scgraphic/results/`
- **Logs**: `/users/ssridh26/scratch/t2_human_scgraphic/logs/`

## Next Steps
1. Expand to all 6 GAGE-seq cell types (add MEP, MLP, B_NK)
2. Integrate GSE253407 Paired Hi-C dataset (~7,500 PBMC cells)
3. Train from scratch on 10k+ cells
4. Evaluate bulk Hi-C pre-training strategies

## Contact
scGrapHiC human adaptation by Srimaan Sridharan
Original scGrapHiC: [rsinghlab/scGrapHiC](https://github.com/rsinghlab/scGrapHiC)
