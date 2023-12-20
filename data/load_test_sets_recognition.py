from os import path
import bcolz
import numpy as np


def get_val_pair(folder_path, name, rgb=True):
    samples = np.load(path.join(folder_path, name + f"/{name}.npy"))
    mean, std = 0.5, 0.5
    if not rgb:
        # convert to bgr for MagFace validation
        samples = samples[:, ::-1, :, :]
        mean, std = 0., 1.
    samples = ((samples / 255.) - mean) / std
    issame = np.loadtxt(path.join(folder_path, name + f"/issame.txt"))
    return samples.astype(np.float32), issame
