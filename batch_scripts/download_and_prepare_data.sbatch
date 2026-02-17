#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --output=/users/ssridh26/scratch/t2_human_scgraphic/logs/download_%j.out
#SBATCH --error=/users/ssridh26/scratch/t2_human_scgraphic/logs/download_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

# =============================================================================
# Download and prepare all data for scGrapHiC human blood inference
# =============================================================================

set -e

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate scgraphic_env

# Create directories
mkdir -p /users/ssridh26/scratch/t2_human_scgraphic/logs
mkdir -p /users/ssridh26/scratch/human_scGrapHiC/GSE238001/scRNA-seq
mkdir -p /users/ssridh26/scratch/human_scGrapHiC/GSE238001/scHi-C
mkdir -p /users/ssridh26/scratch/human_scGrapHiC/annotations
mkdir -p /users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/ctcf
mkdir -p /users/ssridh26/scratch/human_scGrapHiC/annotations/preprocessed/cpg

cd /users/ssridh26/scratch/human_scGrapHiC

echo "============================================"
echo "Step 1: Download GSE238001 data from GEO"
echo "============================================"
echo "Start time: $(date)"

# Download the main TAR file (37.5GB)
if [ ! -f "GSE238001_RAW.tar" ]; then
    echo "Downloading GSE238001_RAW.tar (37.5GB)..."
    wget -c "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE238001&format=file" -O GSE238001_RAW.tar
else
    echo "GSE238001_RAW.tar already exists, skipping download"
fi

# Extract the TAR file
echo "Extracting GSE238001_RAW.tar..."
tar -xvf GSE238001_RAW.tar -C GSE238001/

# Move files to appropriate directories
echo "Organizing files..."
cd GSE238001

# Move scRNA-seq files (*.tsv.gz with RNA in name)
mv *RNA*.tsv.gz scRNA-seq/ 2>/dev/null || true
mv *RNA*.gz scRNA-seq/ 2>/dev/null || true

# Move scHi-C files (*.pairs.gz)
mv *.pairs.gz scHi-C/ 2>/dev/null || true

cd /users/ssridh26/scratch/human_scGrapHiC

echo "============================================"
echo "Step 2: Download GENCODE GTF annotation"
echo "============================================"

if [ ! -f "annotations/gencode.v44.annotation.gtf" ]; then
    echo "Downloading GENCODE v44 GTF..."
    cd annotations
    wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
    gunzip gencode.v44.annotation.gtf.gz
    cd ..
else
    echo "GTF file already exists, skipping"
fi

echo "============================================"
echo "Step 3: Download CTCF ChIP-seq from ENCODE"
echo "============================================"

# Using K562 CTCF ChIP-seq as it's high quality and commonly used
# ENCSR000AKB - CTCF ChIP-seq on K562, hg38
# This provides a good baseline for hematopoietic cells

cd annotations
if [ ! -f "CTCF_hg38.bigWig" ]; then
    echo "Downloading CTCF ChIP-seq signal from ENCODE (K562)..."
    # K562 CTCF fold change over control signal p-value
    wget -c "https://www.encodeproject.org/files/ENCFF796WRU/@@download/ENCFF796WRU.bigWig" -O CTCF_hg38.bigWig
else
    echo "CTCF bigWig already exists, skipping"
fi

echo "============================================"
echo "Step 4: Download CpG Islands from UCSC"
echo "============================================"

if [ ! -f "cpgIslandExt.txt" ]; then
    echo "Downloading CpG Islands from UCSC..."
    wget -c "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz"
    gunzip cpgIslandExt.txt.gz
else
    echo "CpG Islands already exists, skipping"
fi

cd /users/ssridh26/scratch/human_scGrapHiC

echo "============================================"
echo "Step 5: Summary"
echo "============================================"

echo "Downloaded files:"
echo "  scRNA-seq files: $(ls -1 GSE238001/scRNA-seq/*.gz 2>/dev/null | wc -l)"
echo "  scHi-C files: $(ls -1 GSE238001/scHi-C/*.pairs.gz 2>/dev/null | wc -l)"
echo "  GTF: $(ls annotations/*.gtf 2>/dev/null)"
echo "  CTCF: $(ls annotations/CTCF*.bigWig 2>/dev/null)"
echo "  CpG: $(ls annotations/cpgIsland*.txt 2>/dev/null)"

echo ""
echo "Next steps:"
echo "1. Run preprocess_annotations.py to create per-chromosome CTCF/CpG .npy files"
echo "2. Run create_pseudobulk.py to aggregate scHi-C by cell type"
echo "3. Run create_human_npz_v2.py to create the inference NPZ"
echo "4. Run inference.py"

echo ""
echo "End time: $(date)"
echo "============================================"
