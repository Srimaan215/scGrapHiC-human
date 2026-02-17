
import numpy as np
import os
import glob

data_dir = "/users/ssridh26/scratch/t2_human_scgraphic/processed/generated_npz/"
files = glob.glob(os.path.join(data_dir, "*_inference.npz"))

results = []

print(f"Found {len(files)} files.")

for f in files:
    try:
        data = np.load(f, allow_pickle=True)
        # Assuming targets is (N, 1, 128, 128)
        if 'targets' not in data:
            print(f"Skipping {os.path.basename(f)}: 'targets' key not found.")
            continue
            
        targets = data['targets']
        
        # Calculate sparsity
        total_elements = targets.size
        zero_elements = total_elements - np.count_nonzero(targets)
        sparsity = (zero_elements / total_elements) * 100.0
        
        # Calculate average contacts per map
        avg_contacts = np.sum(targets) / targets.shape[0]
        
        name = os.path.basename(f).replace("_inference.npz", "")
        results.append({
            "name": name,
            "sparsity": sparsity,
            "avg_contacts": avg_contacts,
            "path": f
        })
        print(f"{name}: Sparsity={sparsity:.2f}%, Avg Contacts={avg_contacts:.2f}")
        
    except Exception as e:
        print(f"Error processing {f}: {e}")

# Sort by sparsity (ascending -> lowest sparsity/most dense first)
results.sort(key=lambda x: x['sparsity'])

print("\n--- Summary (Lowest Sparsity First) ---")
for i, res in enumerate(results):
    print(f"{i+1}. {res['name']}: {res['sparsity']:.2f}% sparse")

if len(results) >= 2:
    print(f"\nThe 2 lowest sparsity (most dense) cell types are:")
    print(f"1. {results[0]['name']}")
    print(f"2. {results[1]['name']}")
else:
    print("\nNot enough files to pick top 2.")
