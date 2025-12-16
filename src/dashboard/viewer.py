
import numpy as np
from preprocessing.load_nifti import load_mri
from preprocessing.normalize import min_max_normalize

def get_central_slices(mri_path: str, num_slices: int = 9):
    vol = load_mri(mri_path)
    vol = min_max_normalize(vol)
    depth = vol.shape[2]
    center = depth // 2
    half = num_slices // 2
    indices = range(max(center - half, 0), min(center + half, depth))
    slices = [vol[:, :, i] for i in indices]
    return np.stack(slices, axis=0)
