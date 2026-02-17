#!/usr/bin/env python3
"""
Run scGrapHiC inference on human data with proper checkpoint loading.

The checkpoint was saved with node_features=5 (after adding CTCF+CpG),
but when loading we need to pass node_features=2 so the model adds +2+1=3
to get 5 total.
"""

import sys
sys.path.insert(0, '/users/ssridh26/projects/scGrapHiC')

import numpy as np
import torch
from pathlib import Path

# Import model
from src.model import scGrapHiC

def main():
    # Paths
    checkpoint_path = '/oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights/scgraphic.ckpt'
    data_path = '/users/ssridh26/scratch/t2_human_scgraphic/processed/HSC_inference.npz'
    output_path = '/users/ssridh26/scratch/t2_human_scgraphic/results/HSC_inference_results.npz'
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint to get parameters
    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    params = ckpt['hyper_parameters']['PARAMETERS']
    
    print("Checkpoint parameters:")
    print(f"  node_features (saved): {params['node_features']}")
    print(f"  ctcf_motif: {params['ctcf_motif']}")
    print(f"  cpg_motif: {params['cpg_motif']}")
    print(f"  pos_encodings_dim: {params['pos_encodings_dim']}")
    
    # The model adds +2 for CTCF and +1 for CpG during __init__
    # So we need to pass node_features=2 to get 5 after addition
    # node_features=2 (RNA) + 2 (CTCF) + 1 (CpG) = 5
    params['node_features'] = 2  # RNA only, model will add CTCF/CpG
    
    print(f"\nAdjusted node_features for loading: {params['node_features']}")
    print(f"After model adds CTCF(+2) + CpG(+1): {params['node_features'] + 3}")
    
    # Create model with adjusted parameters
    print("\nCreating model...")
    model = scGrapHiC(params)
    
    # Load state dict
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    node_features = data['node_features']  # (N, 128, 5)
    pes = data['pes']  # (N, 128, 16)
    bulk_hics = data['bulk_hics']  # (N, 1, 128, 128)
    targets = data['targets']  # (N, 1, 128, 128)
    indexes = data['indexes']
    metadatas = data['metadatas']
    
    print(f"Data shapes:")
    print(f"  node_features: {node_features.shape}")
    print(f"  pes: {pes.shape}")
    print(f"  bulk_hics: {bulk_hics.shape}")
    print(f"  targets: {targets.shape}")
    
    # Run inference in batches
    batch_size = 32
    n_samples = node_features.shape[0]
    all_outputs = []
    
    print(f"\nRunning inference on {n_samples} samples (batch_size={batch_size})...")
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            
            # Get batch
            nf_batch = torch.from_numpy(node_features[i:end_idx]).float().to(device)
            pe_batch = torch.from_numpy(pes[i:end_idx]).float().to(device)
            bulk_batch = torch.from_numpy(bulk_hics[i:end_idx, 0]).float().to(device)
            
            # Forward pass
            outputs = model.transform(nf_batch, pe_batch, bulk_batch)
            all_outputs.append(outputs.cpu().numpy())
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {end_idx}/{n_samples} samples")
    
    # Concatenate outputs
    outputs = np.concatenate(all_outputs, axis=0)
    print(f"\nOutput shape: {outputs.shape}")
    
    # Compute metrics
    print("\nComputing metrics...")
    from skimage.metrics import structural_similarity as ssim
    from scipy.stats import spearmanr
    
    mse_scores = []
    ssim_scores = []
    scc_scores = []
    
    for i in range(n_samples):
        pred = outputs[i, 0]
        target = targets[i, 0]
        
        # MSE
        mse = np.mean((pred - target) ** 2)
        mse_scores.append(mse)
        
        # SSIM
        data_range = max(target.max() - target.min(), pred.max() - pred.min())
        if data_range > 0:
            s = ssim(target, pred, data_range=data_range)
        else:
            s = 1.0
        ssim_scores.append(s)
        
        # SCC (Spearman)
        corr, _ = spearmanr(pred.flatten(), target.flatten())
        if np.isnan(corr):
            corr = 0.0
        scc_scores.append(corr)
    
    print(f"\nResults:")
    print(f"  MSE:  mean={np.mean(mse_scores):.4f}, std={np.std(mse_scores):.4f}")
    print(f"  SSIM: mean={np.mean(ssim_scores):.4f}, std={np.std(ssim_scores):.4f}")
    print(f"  SCC:  mean={np.mean(scc_scores):.4f}, std={np.std(scc_scores):.4f}")
    
    # Check output statistics
    print(f"\nOutput statistics:")
    print(f"  Generated - min: {outputs.min():.4f}, max: {outputs.max():.4f}, mean: {outputs.mean():.4f}")
    print(f"  Targets   - min: {targets.min():.4f}, max: {targets.max():.4f}, mean: {targets.mean():.4f}")
    print(f"  Generated zeros%: {(outputs == 0).sum() / outputs.size * 100:.2f}%")
    print(f"  Targets zeros%: {(targets == 0).sum() / targets.size * 100:.2f}%")
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    np.savez_compressed(
        output_path,
        outputs=outputs,
        targets=targets,
        indexes=indexes,
        metadatas=metadatas,
        mse_scores=np.array(mse_scores),
        ssim_scores=np.array(ssim_scores),
        scc_scores=np.array(scc_scores),
    )
    
    print("Done!")

if __name__ == '__main__':
    main()
