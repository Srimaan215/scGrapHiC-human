#!/usr/bin/env python3
"""
Fine-tune scGrapHiC on human blood HSC data.

This script loads the pretrained mouse model and fine-tunes it on human data.
Uses the train/val/test splits created by create_train_test_val_split.py.

Usage:
    python finetune_human.py --epochs 100 --lr 1e-5 --experiment human_hsc_finetune

Key differences from original training:
1. Uses pretrained checkpoint instead of random initialization
2. Lower learning rate for fine-tuning
3. Human-specific data splits
4. More frequent validation (every 10 epochs instead of 50)
"""

import os
import torch
import argparse
import numpy as np
import lightning.pytorch as pl
from pathlib import Path

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.globals import RESULTS
from src.model import GenomicDataset, scGrapHiC
from src.utils import create_directory


# Default paths
DEFAULT_CHECKPOINT = "/oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt"
DEFAULT_DATA_DIR = "/users/ssridh26/scratch/t2_human_scgraphic/processed/splits"
DEFAULT_OUTPUT_DIR = "/users/ssridh26/scratch/t2_human_scgraphic/finetuned_weights"


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune scGrapHiC on human data')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Directory containing train/val/test NPZ files')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save fine-tuned weights')
    
    # Training arguments
    parser.add_argument('--experiment', type=str, default='human_hsc_finetune',
                        help='Experiment name for logging')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--val_every', type=int, default=10,
                        help='Validate every N epochs')
    parser.add_argument('--early_stopping', type=int, default=30,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    
    # Model arguments (must match checkpoint)
    parser.add_argument('--node_features', type=int, default=2,
                        help='Number of node features (model adds +3 for CTCF/CpG)')
    parser.add_argument('--pos_encodings_dim', type=int, default=16,
                        help='Positional encoding dimension')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights, only train decoder')
    
    return parser.parse_args()


def get_parameters(args):
    """Create PARAMETERS dict matching the checkpoint configuration.
    
    All parameters must match what the checkpoint was trained with.
    """
    return {
        # Experiment parameters
        'experiment': args.experiment,
        'seed': args.seed,
        
        # Dataset parameters
        'resolution': 50000,
        'library_size': 25000,
        'normalize_umi': False,
        'normalize_track': True,
        'num_cells_cutoff': 190,
        
        # Hi-C normalization parameters
        'normalization_algorithm': 'library_size_normalization',
        'hic_smoothing': True,
        'smoothing_threshold': 0.25,
        
        # Dataset creation parameters
        'bounds': 10,
        'stride': 32,
        'padding': True,
        'num_nodes': 128,
        'remove_borders': 30000000,
        'batch_size': args.batch_size,
        
        # Ablation parameters (MUST match checkpoint exactly!)
        # These values come from the checkpoint's hyper_parameters
        # Note: node_features starts at 2, model adds +2 for CTCF and +1 for CpG = 5 total
        # Then with positional_encodings: 5 + 16 = 21 total channels
        'rna_seq': True,
        'use_bulk': True,
        'positional_encodings': True,
        'ctcf_motif': True,
        'cpg_motif': True,
        'node_features': 2,  # Model adds +3 for CTCF/CpG to get 5
        'pos_encodings_dim': 16,  # Fixed to match checkpoint
        'bulk_hic': 'mesc',
        
        # Model Encoder Parameters
        'conv1d_kernel_size': 16,
        'encoder_hidden_embedding_size': 32,
        'num_encoder_attn_blocks': 4,
        'num_heads_encoder_attn_blocks': 1,
        'num_graph_conv_blocks': 1,
        'num_graph_encoder_blocks': 4,
        'edge_dims': 1,
        
        # Model Decoder parameters
        'num_decoder_residual_blocks': 7,
        'width': 7,
        'num_channels': 1,
        
        # Loss function parameters
        'loss_scale': 1.0,
        
        # Model training Parameters
        'epochs': args.epochs,
        'gradient_clip_value': args.gradient_clip,
    }


def main():
    args = parse_args()
    PARAMETERS = get_parameters(args)
    
    # Set seed
    pl.seed_everything(args.seed)
    
    print("="*60)
    print("SCGRAPHIC HUMAN DATA FINE-TUNING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Freeze encoder: {args.freeze_encoder}")
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Auto-detect split files
    data_dir_path = Path(args.data_dir)
    train_files = list(data_dir_path.glob('*_train.npz'))
    val_files = list(data_dir_path.glob('*_val.npz'))
    test_files = list(data_dir_path.glob('*_test.npz'))
    
    if not train_files:
        raise FileNotFoundError(f"No *_train.npz file found in {args.data_dir}")
    
    train_file = train_files[0]
    val_file = val_files[0] if val_files else None
    test_file = test_files[0] if test_files else None
    
    print(f"\nDetected split files:")
    print(f"  Train: {train_file.name}")
    print(f"  Val: {val_file.name if val_file else 'None'}")
    print(f"  Test: {test_file.name if test_file else 'None'}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = GenomicDataset(str(train_file), PARAMETERS)
    
    if val_file:
        val_dataset = GenomicDataset(str(val_file), PARAMETERS)
    else:
        # Use a portion of training data for validation if no val file
        print("  Warning: No validation file found, using 10% of training data")
        val_size = len(train_dataset) // 10
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [len(train_dataset) - val_size, val_size]
        )
    
    if test_file:
        test_dataset = GenomicDataset(str(test_file), PARAMETERS)
    else:
        print("  Warning: No test file found, using validation set for testing")
        test_dataset = val_dataset
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {args.checkpoint}...")
    
    # Create model with correct parameters
    model = scGrapHiC(PARAMETERS)
    
    # Load pretrained weights
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Pretrained weights loaded successfully")
    
    # Optionally freeze encoder
    if args.freeze_encoder:
        print("Freezing encoder weights...")
        for name, param in model.named_parameters():
            if 'encoder' in name.lower() or 'conv' in name.lower():
                param.requires_grad = False
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,}")
    
    # Override optimizer settings for fine-tuning
    # We'll need to modify the configure_optimizers to use our learning rate
    original_configure_optimizers = model.configure_optimizers
    def new_configure_optimizers():
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=0.01
        )
        # Use StepLR instead of ReduceLROnPlateau to avoid metric availability issues
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    model.configure_optimizers = new_configure_optimizers
    
    # Setup logging and callbacks
    tb_logger = TensorBoardLogger(
        os.path.join(args.output_dir, "logs"),
        name=args.experiment
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='{epoch}-{valid/SCC:.4f}',
        monitor='valid/SCC',
        save_top_k=3,
        mode='max',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='valid/SCC',
        patience=args.early_stopping,
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_every,
        logger=tb_logger,
        deterministic=True,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        gradient_clip_val=args.gradient_clip,
        accelerator='auto',
        devices=1,
    )
    
    # Train
    print("\nStarting fine-tuning...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test with best model
    print("\nTesting with best model...")
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model: {best_model_path}")
    
    trainer.test(model, test_loader, ckpt_path=best_model_path)
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'scgraphic_human_hsc.ckpt')
    trainer.save_checkpoint(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
