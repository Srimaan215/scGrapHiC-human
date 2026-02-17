import os
import sys
import torch
import argparse

import numpy as np
import lightning.pytorch as pl

from src.globals import *
from src.evaluations import evaluate
from src.model import GenomicDataset, scGrapHiC
from src.utils import initialize_parameters_from_args

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


# =============================================================================
# CONFIGURATION
# =============================================================================

# Parse command line arguments
parser = argparse.ArgumentParser(description='scGrapHiC Inference')
parser.add_argument('--dataset', type=str, default='GSE238001', help='Dataset name')
parser.add_argument('--cell-type', type=str, default='HSC', help='Cell type to run inference on')
parser.add_argument('--npz-file', type=str, default=None, help='Path to NPZ file (overrides default)')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (default: MODEL_WEIGHTS/scgraphic.ckpt)')
args, remaining_args = parser.parse_known_args()

# Load parameters from checkpoint to match model architecture
ckpt_path = args.checkpoint if args.checkpoint else os.path.join(MODEL_WEIGHTS, 'scgraphic.ckpt')
checkpoint = torch.load(ckpt_path, map_location='cpu')
PARAMETERS = checkpoint['hyper_parameters']['PARAMETERS']

# IMPORTANT: The checkpoint was trained with node_features=5 (2 RNA + 2 CTCF + 1 CpG)
# The model's __init__ adds +2 for CTCF and +1 for CpG when these flags are True.
# So we need to set node_features=2 (RNA only) and let the model add CTCF+CpG.
#PARAMETERS['node_features'] = 2
PARAMETERS['ctcf_motif'] = False
PARAMETERS['cpg_motif'] = False #set these 2 to true later once I train my own set of human data

pl.seed_everything(PARAMETERS['seed'])

# Set experiment name
PARAMETERS['experiment'] = f"{args.dataset}_{args.cell_type}"

print("=" * 60)
print(f"scGrapHiC Inference - Human Data")
print("=" * 60)
print(f"Dataset: {args.dataset}")
print(f"Cell Type: {args.cell_type}")
print(f"Experiment: {PARAMETERS['experiment']}")
print(PARAMETERS)

# =============================================================================
# LOAD DATASET
# =============================================================================

# Determine NPZ file path
if args.npz_file:
    npz_path = args.npz_file
else:
    npz_path = os.path.join(HUMAN_PROCESSED_DATA_GSE238001, f"{args.cell_type}_inference.npz")

print(f"\nLoading dataset: {npz_path}")

test_dataset = GenomicDataset(npz_path, PARAMETERS)
test_data_loader = torch.utils.data.DataLoader(test_dataset, PARAMETERS['batch_size'], shuffle=False)

print(f"Loaded {len(test_dataset)} windows")

# =============================================================================
# LOAD MODEL
# =============================================================================

tb_logger = TensorBoardLogger("logs", name=PARAMETERS['experiment'])
checkpoint_callback = ModelCheckpoint(monitor="valid/SCC", save_top_k=3, mode='max')
scgraphic = scGrapHiC(PARAMETERS)

# Use GPU if available
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing accelerator: {accelerator}")

trainer = pl.Trainer(
    accelerator=accelerator,
    max_epochs=PARAMETERS['epochs'], 
    check_val_every_n_epoch=50, 
    logger=tb_logger,
    deterministic=True,
    callbacks=[checkpoint_callback],
    gradient_clip_val=PARAMETERS['gradient_clip_value'],
    profiler="simple"
)

ckpt_path = args.checkpoint if args.checkpoint else os.path.join(MODEL_WEIGHTS, 'scgraphic.ckpt')
print(f"Loading checkpoint: {ckpt_path}")

checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
scgraphic.load_state_dict(checkpoint['state_dict'])
print("Checkpoint loaded successfully")

# =============================================================================
# RUN INFERENCE
# =============================================================================

print(f"\nRunning inference...")
trainer.test(scgraphic, test_data_loader)

# =============================================================================
# EVALUATE
# =============================================================================

results_path = os.path.join(RESULTS, PARAMETERS['experiment'])
print(f"\nEvaluating results: {results_path}")
evaluate(results_path, PARAMETERS)

print("\n" + "=" * 60)
print("Inference complete!")
print("=" * 60)
