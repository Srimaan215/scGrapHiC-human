
import numpy as np

data_path = "/users/ssridh26/data/ssridh26/t2_human_scgraphic/processed/HSC_inference.npz"
try:
    data = np.load(data_path, allow_pickle=True)
    print("Keys:", data.files)
    for key in data.files:
        val = data[key]
        if isinstance(val, np.ndarray):
            print(f"{key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"{key}: {type(val)}")
except FileNotFoundError:
    print(f"File not found: {data_path}")
    # try the scratch one
    data_path = "/users/ssridh26/scratch/t2_human_scgraphic/processed/HSC_inference.npz"
    try:
        data = np.load(data_path, allow_pickle=True)
        print(f"Found at {data_path}")
        print("Keys:", data.files)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
