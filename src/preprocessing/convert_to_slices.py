
import os
from typing import Optional
import numpy as np
from PIL import Image

from src.preprocessing.load_nifti import load_mri
from src.preprocessing.normalize import min_max_normalize

def volume_to_slices(
    mri_path: str,
    output_dir: str,
    num_slices: int = 32,
    axis: int = 2,
    resize: Optional[tuple] = (128, 128)
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    vol = load_mri(mri_path)
    vol = min_max_normalize(vol)

    vol = np.moveaxis(vol, axis, -1)
    depth = vol.shape[-1]

    center = depth // 2
    half = num_slices // 2
    start = max(center - half, 0)
    end = min(center + half, depth)

    for i, idx in enumerate(range(start, end)):
        slice_img = vol[:, :, idx]
        slice_img = (slice_img * 255).astype(np.uint8)
        im = Image.fromarray(slice_img, mode="L")
        if resize is not None:
            im = im.resize(resize)
        im.save(os.path.join(output_dir, f"slice_{i:03d}.png"))
