#!/usr/bin/env python3
"""
Verify NPZ files are correctly formatted for scGrapHiC inference.
Compares dimensions and data ranges with a known-good file (HSC).
"""
import numpy as np
from pathlib import Path
import sys

def check_npz_file(npz_path, reference=None):
    """Check NPZ file structure and dimensions."""
    print(f"\n{'='*70}")
    print(f"Checking: {npz_path.name}")
    print(f"{'='*70}")
    
    if not npz_path.exists():
        print(f"❌ File not found!")
        return False
    
    # File size
    size_mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    # Check required keys
    required_keys = ['node_features', 'targets', 'pes', 'bulk_hics', 'indexes', 'metadatas']
    print(f"\n📋 Keys present: {list(data.keys())}")
    
    missing = [k for k in required_keys if k not in data.keys()]
    if missing:
        print(f"❌ Missing required keys: {missing}")
        return False
    print("✅ All required keys present")
    
    # Check shapes
    print(f"\n📐 Dimensions:")
    shapes = {}
    for key in required_keys:
        arr = data[key]
        shapes[key] = arr.shape
        print(f"  {key:15s}: {arr.shape}")
    
    # Verify consistent N (number of samples)
    n_samples = shapes['node_features'][0]
    inconsistent = []
    for key in ['targets', 'pes', 'bulk_hics', 'indexes', 'metadatas']:
        if shapes[key][0] != n_samples:
            inconsistent.append(f"{key} has {shapes[key][0]} samples")
    
    if inconsistent:
        print(f"\n❌ Inconsistent sample counts:")
        for msg in inconsistent:
            print(f"  - {msg}")
        return False
    print(f"✅ Consistent sample count: {n_samples}")
    
    # Check expected dimensions
    print(f"\n🔍 Dimension validation:")
    checks_passed = True
    
    # node_features should be (N, 128, 5)
    if shapes['node_features'] != (n_samples, 128, 5):
        print(f"  ❌ node_features: Expected (N, 128, 5), got {shapes['node_features']}")
        checks_passed = False
    else:
        print(f"  ✅ node_features: (N, 128, 5)")
    
    # targets should be (N, 1, 128, 128)
    if shapes['targets'] != (n_samples, 1, 128, 128):
        print(f"  ❌ targets: Expected (N, 1, 128, 128), got {shapes['targets']}")
        checks_passed = False
    else:
        print(f"  ✅ targets: (N, 1, 128, 128)")
    
    # pes should be (N, 128, 16)
    if shapes['pes'] != (n_samples, 128, 16):
        print(f"  ❌ pes: Expected (N, 128, 16), got {shapes['pes']}")
        checks_passed = False
    else:
        print(f"  ✅ pes: (N, 128, 16)")
    
    # bulk_hics should be (N, 1, 128, 128)
    if shapes['bulk_hics'] != (n_samples, 1, 128, 128):
        print(f"  ❌ bulk_hics: Expected (N, 1, 128, 128), got {shapes['bulk_hics']}")
        checks_passed = False
    else:
        print(f"  ✅ bulk_hics: (N, 1, 128, 128)")
    
    # indexes should be (N, 4)
    if shapes['indexes'] != (n_samples, 4):
        print(f"  ❌ indexes: Expected (N, 4), got {shapes['indexes']}")
        checks_passed = False
    else:
        print(f"  ✅ indexes: (N, 4)")
    
    # metadatas should be (N, 4)
    if shapes['metadatas'] != (n_samples, 4):
        print(f"  ❌ metadatas: Expected (N, 4), got {shapes['metadatas']}")
        checks_passed = False
    else:
        print(f"  ✅ metadatas: (N, 4)")
    
    # Check data ranges
    print(f"\n📊 Data ranges:")
    node_features = data['node_features']
    targets = data['targets']
    pes = data['pes']
    
    print(f"  node_features:")
    for ch in range(5):
        ch_data = node_features[:, :, ch]
        print(f"    Ch {ch}: [{ch_data.min():.4f}, {ch_data.max():.4f}], mean={ch_data.mean():.4f}")
    
    print(f"  targets: [{targets.min():.4f}, {targets.max():.4f}], mean={targets.mean():.4f}")
    print(f"  pes: [{pes.min():.4f}, {pes.max():.4f}], mean={pes.mean():.4f}")
    
    # Check for NaN/Inf
    print(f"\n🔬 Data quality:")
    issues = []
    for key in ['node_features', 'targets', 'pes', 'bulk_hics']:
        arr = data[key]
        if np.isnan(arr).any():
            issues.append(f"{key} contains NaN values")
        if np.isinf(arr).any():
            issues.append(f"{key} contains Inf values")
    
    if issues:
        print(f"  ❌ Issues found:")
        for issue in issues:
            print(f"    - {issue}")
        checks_passed = False
    else:
        print(f"  ✅ No NaN or Inf values")
    
    # Compare with reference if provided
    if reference:
        print(f"\n🔄 Comparison with reference (HSC):")
        try:
            ref_data = np.load(reference, allow_pickle=True)
            
            # Compare shapes
            if data['node_features'].shape[1:] == ref_data['node_features'].shape[1:]:
                print(f"  ✅ Feature shapes match reference")
            else:
                print(f"  ⚠️  Feature shapes differ from reference")
            
            # Compare metadata structure
            if data['metadatas'].shape[1] == ref_data['metadatas'].shape[1]:
                print(f"  ✅ Metadata structure matches reference")
            else:
                print(f"  ⚠️  Metadata structure differs from reference")
                
            # Compare sample counts (approximately)
            ratio = data['node_features'].shape[0] / ref_data['node_features'].shape[0]
            print(f"  Sample count ratio (this/ref): {ratio:.2f}x")
            
        except Exception as e:
            print(f"  ⚠️  Could not compare with reference: {e}")
    
    # Final verdict
    print(f"\n{'='*70}")
    if checks_passed:
        print(f"✅ {npz_path.name} is properly formatted")
    else:
        print(f"⚠️  {npz_path.name} has some issues (see above)")
    print(f"{'='*70}")
    
    return checks_passed

def main():
    processed_dir = Path("/users/ssridh26/scratch/t2_human_scgraphic/processed")
    
    # Reference file (known good)
    hsc_file = processed_dir / "HSC_inference.npz"
    
    # Files to check
    files_to_check = [
        processed_dir / "MPP_inference.npz",
        processed_dir / "LMPP_inference.npz",
    ]
    
    print("="*70)
    print("NPZ FILE VERIFICATION")
    print("="*70)
    print(f"Reference file: {hsc_file.name}")
    
    # First verify reference exists
    if not hsc_file.exists():
        print(f"\n❌ Reference file not found: {hsc_file}")
        print("Cannot perform comparison.")
        reference = None
    else:
        print(f"✅ Reference file found")
        reference = hsc_file
        # Quick check of reference
        check_npz_file(hsc_file)
    
    # Check each file
    all_passed = True
    for npz_file in files_to_check:
        passed = check_npz_file(npz_file, reference=reference)
        all_passed = all_passed and passed
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for npz_file in files_to_check:
        status = "✅" if npz_file.exists() else "❌ NOT FOUND"
        print(f"{status} {npz_file.name}")
    
    if all_passed:
        print(f"\n🎉 All files are properly formatted and ready for inference!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some files have issues. Please review above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
