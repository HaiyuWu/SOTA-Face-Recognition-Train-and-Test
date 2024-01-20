from os import path
import numpy as np


def get_val_pair(folder_path, name, rgb=True, mean=0.5, std=0.5):
    samples = np.load(path.join(folder_path, name + f"/{name}.npy"))
    if not rgb:
        # convert to bgr for MagFace validation
        samples = samples[:, ::-1, :, :]
    samples = ((samples / 255.) - mean) / std
    issame = np.loadtxt(path.join(folder_path, name + f"/issame.txt"))
    # add horizontal flip test
    samples = np.vstack((samples, samples[..., ::-1]))
    return samples.astype(np.float32), issame
