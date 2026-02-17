#!/usr/bin/env python3
"""
Download bulk Hi-C files for human from multiple sources.
Tries multiple mirrors and sources until one succeeds.
"""
import os
import sys
import requests
import subprocess
from pathlib import Path

def download_with_wget(url, output_file, desc=""):
    """Download using wget with progress bar"""
    print(f"\n{'='*70}")
    print(f"Attempting: {desc}")
    print(f"URL: {url}")
    print(f"Output: {output_file}")
    print(f"{'='*70}")
    
    cmd = ['wget', '--no-check-certificate', '--show-progress', '-O', output_file, url]
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0 and os.path.exists(output_file):
        size = os.path.getsize(output_file)
        if size > 1_000_000:  # > 1MB
            print(f"✅ Success! Downloaded {size / 1e9:.2f} GB")
            return True
        else:
            print(f"❌ Failed: File too small ({size} bytes)")
            os.remove(output_file)
            return False
    else:
        print(f"❌ Failed: Download error")
        if os.path.exists(output_file):
            os.remove(output_file)
        return False


def verify_hic_file(filepath):
    """Verify .hic file is valid"""
    try:
        import hicstraw
        hic = hicstraw.HiCFile(filepath)
        chroms = [c.name for c in hic.getChromosomes()[:3]]
        resolutions = hic.getResolutions()
        
        # Check if it has 50kb resolution
        has_50kb = 50000 in resolutions
        
        print(f"\n✓ File validation:")
        print(f"  Chromosomes: {chroms}")
        print(f"  Resolutions: {resolutions[:5]}")
        print(f"  Has 50kb: {has_50kb}")
        
        return len(chroms) > 0 and has_50kb
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def main():
    output_dir = Path("/users/ssridh26")
    
    # Source options for CD34+ and GM12878
    sources = [
        # ENCODE portal - CD34+
        {
            'name': 'CD34_mobilized_ENCODE',
            'output': output_dir / 'CD34_mobilized_ENCODE.hic',
            'url': 'https://www.encodeproject.org/files/ENCFF851JZJ/@@download/ENCFF851JZJ.hic',
            'desc': 'CD34+ mobilized from ENCODE (ENCFF851JZJ)'
        },
        # ENCODE portal - GM12878
        {
            'name': 'GM12878_ENCODE',
            'output': output_dir / 'GM12878_ENCODE.hic',
            'url': 'https://www.encodeproject.org/files/ENCFF718AWL/@@download/ENCFF718AWL.hic',
            'desc': 'GM12878 from ENCODE (ENCFF718AWL)'
        },
        # Alternative ENCODE GM12878
        {
            'name': 'GM12878_ENCODE_alt',
            'output': output_dir / 'GM12878_ENCODE_alt.hic',
            'url': 'https://www.encodeproject.org/files/ENCFF336QYL/@@download/ENCFF336QYL.hic',
            'desc': 'GM12878 from ENCODE alt (ENCFF336QYL)'
        },
        # 4DN portal - alternative IDs
        {
            'name': 'GM12878_4DN',
            'output': output_dir / 'GM12878_4DN.hic',
            'url': 'https://data.4dnucleome.org/files-processed/4DNFI1UEG1HD/@@download/4DNFI1UEG1HD.hic',
            'desc': 'GM12878 from 4DN (4DNFI1UEG1HD)'
        },
    ]
    
    print("\n" + "="*70)
    print("BULK HI-C DOWNLOAD UTILITY")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Trying {len(sources)} sources...")
    
    successful_downloads = []
    
    for source in sources:
        if download_with_wget(source['url'], str(source['output']), source['desc']):
            if verify_hic_file(str(source['output'])):
                successful_downloads.append(source)
                print(f"\n✅ VALID: {source['name']}")
            else:
                print(f"\n❌ INVALID: {source['name']} (not a valid .hic file)")
                os.remove(source['output'])
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if successful_downloads:
        print(f"Successfully downloaded {len(successful_downloads)} file(s):")
        for source in successful_downloads:
            size = os.path.getsize(source['output']) / 1e9
            print(f"  ✓ {source['name']}: {size:.2f} GB")
            print(f"    Path: {source['output']}")
        
        print("\nRecommendation:")
        # Prefer CD34+ if available, otherwise GM12878
        cd34_files = [s for s in successful_downloads if 'CD34' in s['name']]
        if cd34_files:
            print(f"  Use CD34+: {cd34_files[0]['output']}")
            print(f"  (Best biological match for HSC/MPP/LMPP)")
        else:
            print(f"  Use GM12878: {successful_downloads[0]['output']}")
            print(f"  (B-lymphocyte, blood lineage)")
            
    else:
        print("❌ No files successfully downloaded.")
        print("\nAlternative options:")
        print("1. Use existing K562: /users/ssridh26/4DNFI5IAH9H1.hic (1.6 GB)")
        print("2. Manual download from:")
        print("   - ENCODE: https://www.encodeproject.org/")
        print("   - GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525")
        print("   - 4DN: https://data.4dnucleome.org/")
        
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
