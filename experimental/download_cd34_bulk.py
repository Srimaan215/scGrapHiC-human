#!/usr/bin/env python3
"""
Download and process CD34+ bulk Hi-C from 4DN to replace K562.

CD34+ cells are hematopoietic progenitors - the exact population
containing HSC/MPP/LMPP. This is a much better match than K562 (cancer).

4DN File IDs:
- 4DNFI6T67EPW: CD34+ mobilized (Rao 2014) - High quality
- 4DNFIPK7ZN1V: CD34+ primary cells - Alternative

This script downloads the .hic file and verifies it can be read.
"""

import subprocess
import sys
from pathlib import Path

# 4DN accession IDs
CD34_FILES = {
    '4DNFI6T67EPW': {
        'name': 'CD34_mobilized_Rao2014',
        'url': 'https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/31bba7d8-6a8f-4e8b-be30-e84d37c2c4b4/4DNFI6T67EPW.hic',
        'description': 'CD34+ mobilized cells, high depth'
    },
    '4DNFIPK7ZN1V': {
        'name': 'CD34_primary',
        'url': 'https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/2a268a31-21cb-4c76-89ab-01b5e1bbd79d/4DNFIPK7ZN1V.hic',
        'description': 'CD34+ primary cells'
    }
}

# Alternative: GM12878 (most common reference)
GM12878_FILE = {
    '4DNFI1UEG1HD': {
        'name': 'GM12878_Rao2014',
        'url': 'https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/366caa40-84c3-4a7a-8a1e-c97f7288af83/4DNFI1UEG1HD.hic',
        'description': 'GM12878 B-lymphocyte, gold standard'
    }
}


def download_file(file_id, file_info, output_dir):
    """Download .hic file from 4DN"""
    
    output_path = Path(output_dir) / f"{file_info['name']}.hic"
    
    print(f"Downloading {file_id}: {file_info['description']}")
    print(f"URL: {file_info['url']}")
    print(f"Output: {output_path}")
    print()
    
    if output_path.exists():
        print(f"✓ File already exists: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1e9:.2f} GB")
        return output_path
    
    # Download with wget (shows progress)
    cmd = ['wget', '-O', str(output_path), file_info['url']]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Downloaded successfully!")
        print(f"  Size: {output_path.stat().st_size / 1e9:.2f} GB")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed: {e}")
        return None


def verify_hic_file(hic_file):
    """Verify .hic file can be read with hicstraw"""
    print(f"\nVerifying {hic_file}...")
    
    try:
        import hicstraw
        hic = hicstraw.HiCFile(str(hic_file))
        
        # Get chromosomes
        chroms = hic.getChromosomes()
        print(f"✓ File valid!")
        print(f"  Genome: {hic.getGenomeID()}")
        print(f"  Chromosomes: {len(chroms)}")
        print(f"  Resolutions: {hic.getResolutions()}")
        
        # Test extraction
        test = hic.getMatrixZoomData('chr1', 'chr1', 'observed', 'KR', 'BP', 50000)
        print(f"  Test extraction: Success (chr1 at 50kb)")
        
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    output_dir = Path('/users/ssridh26')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DOWNLOAD BETTER BULK Hi-C FOR HUMAN HEMATOPOIETIC CELLS")
    print("=" * 70)
    print()
    print("Current: K562 (cancer cell line) - POOR MATCH")
    print("Recommended: CD34+ (hematopoietic progenitors) - PERFECT MATCH")
    print()
    
    # Download CD34+ (recommended)
    print("Downloading CD34+ bulk Hi-C...")
    print("-" * 70)
    
    for file_id, file_info in CD34_FILES.items():
        hic_path = download_file(file_id, file_info, output_dir)
        if hic_path:
            verify_hic_file(hic_path)
        print()
        break  # Download just the first one (Rao 2014 is best)
    
    # Optional: Also download GM12878
    print("\n" + "=" * 70)
    print("OPTIONAL: Download GM12878 (gold standard reference)")
    print("=" * 70)
    response = input("\nDownload GM12878 as well? (y/n): ")
    
    if response.lower() == 'y':
        for file_id, file_info in GM12878_FILE.items():
            hic_path = download_file(file_id, file_info, output_dir)
            if hic_path:
                verify_hic_file(hic_path)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Update create_human_npz_v2.py to use new bulk Hi-C:")
    print("   BULK_HIC_FILE = Path('/users/ssridh26/CD34_mobilized_Rao2014.hic')")
    print()
    print("2. Regenerate NPZ files:")
    print("   python create_human_npz_v2.py --cell_type HSC")
    print("   python create_human_npz_v2.py --cell_type MPP")
    print("   python create_human_npz_v2.py --cell_type LMPP")
    print()
    print("3. Re-run check_bulk_mismatch.py")
    print("   Expected: Much fewer mismatches (2-3% vs 5-6% with K562)")
    print()
    print("4. Fine-tune with new data")


if __name__ == '__main__':
    main()
