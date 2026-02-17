#!/usr/bin/env python3
"""
Run full inference across ALL chromosomes for GM12878 fine-tuned model.

This uses the full *_inference.npz files (not just test split) to get
comprehensive results across all chr1-chr22.
"""

import os
import sys
import torch
import argparse
import numpy as np
import lightning.pytorch as pl
from pathlib import Path

from src.globals import *
from src.evaluations import evaluate
from src.model import GenomicDataset, scGrapHiC

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def main():
    parser = argparse.ArgumentParser(description='Run full inference on all chromosomes')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned checkpoint')
    parser.add_argument('--cell_types', nargs='+', default=['HSC', 'MPP', 'LMPP'],
                        help='Cell types to run inference on')
    parser.add_argument('--input_dir', type=str, 
                        default='/users/ssridh26/scratch/t2_human_scgraphic/processed',
                        help='Directory containing *_inference.npz files')
    parser.add_argument('--use_gm12878', action='store_true',
                        help='Use GM12878 regenerated files (default: K562 files)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--experiment_name', type=str, default='human_gm12878_full_inference')
    args = parser.parse_args()

    # Load checkpoint parameters
    print("=" * 60)
    print("Loading checkpoint...")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    PARAMETERS = checkpoint['hyper_parameters']['PARAMETERS'].copy()
    
    # Store batch_size separately for dataloader
    batch_size = args.batch_size
    
    # Set seed
    pl.seed_everything(PARAMETERS.get('seed', 42))
    
    # Create model with checkpoint parameters
    scgraphic = scGrapHiC(PARAMETERS)
    
    # Load weights with strict=False to handle any dimension mismatches
    scgraphic.load_state_dict(checkpoint['state_dict'], strict=False)
    
    print(f"✓ Loaded checkpoint: {args.checkpoint}")
    print(f"  node_features: {PARAMETERS.get('node_features')}")
    print(f"  ctcf_motif: {PARAMETERS.get('ctcf_motif')}")
    print(f"  cpg_motif: {PARAMETERS.get('cpg_motif')}")
    
    # Setup trainer
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Using accelerator: {accelerator}")
    
    # Run inference for each cell type
    for cell_type in args.cell_types:
        print("\n" + "=" * 60)
        print(f"RUNNING INFERENCE: {cell_type}")
        print("=" * 60)
        
        # Determine NPZ file path
        if args.use_gm12878:
            # New GM12878 files (need to be generated first)
            npz_path = os.path.join(args.input_dir, f"{cell_type}_inference_gm12878.npz")
        else:
            # Old K562 files (for comparison)
            npz_path = os.path.join(args.input_dir, f"{cell_type}_inference.npz")
        
        if not os.path.exists(npz_path):
            print(f"❌ File not found: {npz_path}")
            print(f"   Please generate it first using create_human_npz_v2.py")
            continue
        
        print(f"Loading dataset: {npz_path}")
        dataset = GenomicDataset(npz_path, PARAMETERS)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
        
        print(f"Loaded {len(dataset)} windows across all chromosomes")
        
        # Check chromosomes
        meta = np.load(npz_path)['metadatas']
        chroms = sorted(set(meta[:, 3]))
        print(f"Chromosomes: {chroms}")
        
        # Setup experiment name
        PARAMETERS['experiment'] = f"{args.experiment_name}_{cell_type}"
        
        # Create trainer for this cell type
        tb_logger = TensorBoardLogger("logs", name=PARAMETERS['experiment'])
        checkpoint_callback = ModelCheckpoint(monitor="valid/SCC", save_top_k=3, mode='max')
        
        trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=1,  # Just inference
            logger=tb_logger,
            deterministic=True,
            callbacks=[checkpoint_callback],
            enable_checkpointing=False,  # Don't save during inference
        )
        
        # Run inference
        print(f"\nRunning inference...")
        trainer.test(scgraphic, dataloader)
        
        # Evaluate results
        results_path = os.path.join(RESULTS, PARAMETERS['experiment'])
        print(f"\nEvaluating results...")
        evaluate(results_path, PARAMETERS)
        
        # Calculate average SCC
        import pandas as pd
        results_csv = os.path.join(results_path, 'results.csv')
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            avg_scc = df.iloc[:, -1].mean()  # Last column is SCC
            print(f"\n{'='*60}")
            print(f"RESULTS: {cell_type}")
            print(f"  Samples: {len(df)}")
            print(f"  Average SCC: {avg_scc:.4f} ({avg_scc*100:.2f}%)")
            print(f"  Results: {results_csv}")
            print(f"{'='*60}")
    
    print("\n" + "=" * 60)
    print("FULL INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
