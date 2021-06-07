"""
Add wind speed as an extra channel in input data.
"""

import numpy as np
import h5py


def add_wind_speed(data_dir, data_aug_dir):
    with h5py.File(data_dir) as f, h5py.File(data_aug_dir) as new_f:
        data = f["data"][:]
        new_data = np.concatenate(
            [data, np.sqrt(np.power(data[:, :, -2:], 2).sum(axis=2, keepdims=True))],
            axis=2,
        )
        assert new_data.shape == (8736, 12, 6, 15, 14), new_data.shape
        new_f["data"] = new_data
        new_f["mean"] = new_data.mean(axis=(0, 3, 4))
        new_f["std"] = new_data.std(axis=(0, 3, 4))
        new_f["longitude"] = f["longitude"][:]
        new_f["latitude"] = f["latitude"][:]
        new_f["altitude"] = f["altitude"][:]
