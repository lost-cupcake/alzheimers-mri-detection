import numpy as np

def min_max_normalize(volume: np.ndarray) -> np.ndarray:
    vmin = float(volume.min())
    vmax = float(volume.max())

    # If invalid or constant volume → return zeros
    if abs(vmax - vmin) < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)

    volume = (volume - vmin) / (vmax - vmin)
    
    return volume.astype(np.float32)
