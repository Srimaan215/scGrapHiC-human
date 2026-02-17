import os, sys
import numpy as np
# Add project root to path
sys.path.insert(0, '/users/ssridh26/projects/t2_human_scgraphic')

from src.evaluations import evaluate

results_path = '/users/ssridh26/projects/t2_human_scgraphic/results/GSE238001_HSC'

# Remove old result files to avoid duplicate appends
for fn in ['full_results.csv', 'results.csv']:
    fp = os.path.join(results_path, fn)
    if os.path.exists(fp):
        os.remove(fp)
        print(f'Removed {fp}')

# Clean up old chromosight outputs to force re-run with new parameters
print("Cleaning up old chromosight output files...")
for root, dirs, files in os.walk(results_path):
    for file in files:
        if file.endswith("_borders.tsv"):
            os.remove(os.path.join(root, file))

PARAMETERS = {'resolution': 50000}
print('Running evaluate...')
evaluate(results_path, PARAMETERS)
print('Evaluate finished')
