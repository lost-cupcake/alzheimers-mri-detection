import nibabel as nib
import numpy as np
import os

def load_mri(path: str) -> np.ndarray:
    """
    Load NIfTI (.nii / .nii.gz) OR ANALYZE (.hdr/.img) file correctly.
    Ensures the output is a proper 3D numpy array (H, W, D).
    """

    # Handle .hdr case: ensure .img exists
    if path.endswith(".hdr"):
        img_path = path.replace(".hdr", ".img")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing paired .img file for {path}")

    # Load file using nibabel
    img = nib.load(path)
    data = img.get_fdata()

    # FIX 1: ANALYZE format often loads as 4D with last dimension = 1 
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]  # squeeze last dimension

    # FIX 2: ensure it is 3D
    if data.ndim != 3:
        raise ValueError(f"Unexpected MRI shape {data.shape} for file {path}")

    # Convert to float32
    data = np.asarray(data, dtype=np.float32)

    print(f"[INFO] Loaded MRI: {path}, shape={data.shape}")
    return data
