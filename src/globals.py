'''
    This file contains all the static definitions I plan to use in the project
    
'''
import os

# =============================================================================
# BASE PATHS
# =============================================================================
DATA = '/users/ssridh26/scratch/t2_human_scgraphic/'
RAW_DATA = os.path.join(DATA, 'raw')
PREPROCESSED_DATA = os.path.join(DATA, 'preprocessed')
PROCESSED_DATA = os.path.join(DATA, 'processed')

# =============================================================================
# MOUSE (mm10) - Original scGrapHiC paths (kept for reference)
# =============================================================================
MOUSE_RAW_DATA = os.path.join(RAW_DATA, 'mm10')
MOUSE_PREPROCESSED_DATA = os.path.join(PREPROCESSED_DATA, 'mm10')
MOUSE_PROCESSED_DATA = os.path.join(PROCESSED_DATA, 'mm10')

MOUSE_RAW_DATA_HIRES = os.path.join(MOUSE_RAW_DATA, 'HiRES')
MOUSE_RAW_DATA_SCHIC = os.path.join(MOUSE_RAW_DATA_HIRES, 'scHi-C')
MOUSE_RAW_DATA_SCRNASEQ = os.path.join(MOUSE_RAW_DATA_HIRES, 'scRNA-seq')
MOUSE_RAW_BULK_DATA = os.path.join(MOUSE_RAW_DATA, 'bulk')
MOUSE_RAW_MOTIFS_DATA = os.path.join(MOUSE_RAW_DATA, 'motifs')

MOUSE_RAW_DATA_PSEUDO_BULK = os.path.join(MOUSE_RAW_DATA_HIRES, 'pseudo-bulk')
MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ = os.path.join(MOUSE_RAW_DATA_PSEUDO_BULK, 'scRNA-seq')
MOUSE_RAW_DATA_PSEUDO_BULK_SCHIC = os.path.join(MOUSE_RAW_DATA_PSEUDO_BULK, 'scHi-C')

MOUSE_PREPROCESSED_DATA_HIRES = os.path.join(MOUSE_PREPROCESSED_DATA, 'HiRES')
MOUSE_PREPROCESSED_DATA_SCHIC = os.path.join(MOUSE_PREPROCESSED_DATA_HIRES, 'scHi-C')
MOUSE_PREPROCESSED_DATA_SCRNASEQ = os.path.join(MOUSE_PREPROCESSED_DATA_HIRES, 'scRNA-seq')

MOUSE_PREPROCESSED_DATA_PSEUDO_BULK = os.path.join(MOUSE_PREPROCESSED_DATA_HIRES, 'pseudo-bulk')
MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCRNASEQ = os.path.join(MOUSE_PREPROCESSED_DATA_PSEUDO_BULK, 'scRNA-seq')
MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCHIC = os.path.join(MOUSE_PREPROCESSED_DATA_PSEUDO_BULK, 'scHi-C')

MOUSE_PREPROCESSED_DATA_BULK = os.path.join(MOUSE_PREPROCESSED_DATA, 'bulk')
MOUSE_PREPROCESSED_MOTIFS_DATA = os.path.join(MOUSE_PREPROCESSED_DATA, 'motifs')

MOUSE_PROCESSED_DATA_HIRES = os.path.join(MOUSE_PROCESSED_DATA, 'HiRES')

# =============================================================================
# HUMAN (hg38) - GSE238001 GAGE-seq dataset
# =============================================================================
HUMAN_RAW_DATA = os.path.join(RAW_DATA, 'hg38')
HUMAN_PREPROCESSED_DATA = os.path.join(PREPROCESSED_DATA, 'hg38')
HUMAN_PROCESSED_DATA = os.path.join(PROCESSED_DATA, 'hg38')

# Raw data paths
HUMAN_RAW_DATA_GSE238001 = os.path.join(HUMAN_RAW_DATA, 'GSE238001')
HUMAN_RAW_DATA_SCHIC = os.path.join(HUMAN_RAW_DATA_GSE238001, 'scHi-C')
HUMAN_RAW_DATA_SCRNASEQ = os.path.join(HUMAN_RAW_DATA_GSE238001, 'scRNA-seq')
HUMAN_RAW_BULK_DATA = os.path.join(HUMAN_RAW_DATA, 'bulk')
HUMAN_RAW_MOTIFS_DATA = os.path.join(HUMAN_RAW_DATA, 'motifs')

# Preprocessed data paths
HUMAN_PREPROCESSED_DATA_GSE238001 = os.path.join(HUMAN_PREPROCESSED_DATA, 'GSE238001')
HUMAN_PREPROCESSED_DATA_SCHIC = os.path.join(HUMAN_PREPROCESSED_DATA_GSE238001, 'scHi-C')
HUMAN_PREPROCESSED_DATA_SCRNASEQ = os.path.join(HUMAN_PREPROCESSED_DATA_GSE238001, 'scRNA-seq')

HUMAN_PREPROCESSED_DATA_PSEUDO_BULK = os.path.join(HUMAN_PREPROCESSED_DATA_GSE238001, 'pseudo-bulk')
HUMAN_PREPROCESSED_DATA_PSEUDO_BULK_SCRNASEQ = os.path.join(HUMAN_PREPROCESSED_DATA_PSEUDO_BULK, 'scRNA-seq')
HUMAN_PREPROCESSED_DATA_PSEUDO_BULK_SCHIC = os.path.join(HUMAN_PREPROCESSED_DATA_PSEUDO_BULK, 'scHi-C')

HUMAN_PREPROCESSED_DATA_BULK = os.path.join(HUMAN_PREPROCESSED_DATA, 'bulk')
HUMAN_PREPROCESSED_MOTIFS_DATA = os.path.join(HUMAN_PREPROCESSED_DATA, 'motifs')

# Processed data - directly under DATA/processed (not hg38 subdirectory)
HUMAN_PROCESSED_DATA_GSE238001 = os.path.join(DATA, 'processed')

# Human genome annotation
HG38_GTF3_FILE_PATH = os.path.join(HUMAN_RAW_DATA, 'gencode.v44.annotation.gff3.gz')
HUMAN_GENE_COORDINATES_FILE = os.path.join(HUMAN_PREPROCESSED_DATA, 'gene_coordinates.csv')

# Human bulk Hi-C (K562)
HUMAN_K562_BULK_HIC = os.path.join(HUMAN_RAW_BULK_DATA, 'K562_bulk.hic')

# =============================================================================
# MOUSE METADATA (original)
# =============================================================================
HIRES_SERIES_MATRIX_FILE = os.path.join(MOUSE_RAW_DATA_HIRES,'GSE223917_series_matrix.txt')
HIRES_BRAIN_METADATA_FILE = os.path.join(MOUSE_RAW_DATA_HIRES, 'metadata', 'brain_metadata.xlsx')
HIRES_EMBRYO_METADATA_FILE = os.path.join(MOUSE_RAW_DATA_HIRES, 'metadata', 'embryo_metadata.xlsx')

MM10_GTF3_FILE_PATH = os.path.join(MOUSE_RAW_DATA, 'gencode.vM23.annotation.gff3.gz')

# BULK HIC DATASETS (Mouse)
MOUSE_PN5_ZYGOTE_BULK_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'pn5_zygote.hic')
MOUSE_EARLY_TWO_CELL_BULK_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'early_two_cell.hic')
MOUSE_LATE_TWO_CELL_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'late_two_cell.hic')
MOUSE_EIGHT_CELL_BULK_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'eight_cells.hic')
MOUSE_INNER_CELL_MASS_BULK_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'inner_cell_mass.hic')
MOUSE_MESC_BULK_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'mesc.hic')
MOUSE_CEREBRAL_CORETEX_BULK_HIC = os.path.join(MOUSE_RAW_BULK_DATA, 'cerebral_cortex.hic')

# =============================================================================
# MODEL WEIGHTS AND RESULTS
# =============================================================================
MODEL_WEIGHTS = '/oscar/data/rsingh47/ssridh26/scGrapHiC_data/weights'
RESULTS = '/users/ssridh26/projects/t2_human_scgraphic/results'
DATASET_LABELS_JSON = '/users/ssridh26/projects/t2_human_scgraphic/dataset_labels.json'

