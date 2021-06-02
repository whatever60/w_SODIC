import itertools

import numpy as np
import h5py
import cv2
import torch
from torch.utils.data import Dataset


class SuperResoNC(Dataset):
    def __init__(
        self,
        data_dir: str,
        means: np.ndarray,
        stds: np.ndarray,
        *,
        use_cv2=False,
        input_size: tuple = None,  # used when `use_cv2` is True.
        step: tuple = None,  # used when `use_cv2` is False.
        shift: tuple = (0, 0),  # used when `use_cv2` is False
    ):
        with h5py.File(data_dir) as f:
            self.data = f["data"][:]
        self.height = self.data.shape[1]
        self.data -= means.reshape(1, *means.shape, 1, 1)
        self.data /= stds.reshape(1, *means.shape, 1, 1)

        self.use_cv2 = use_cv2
        if use_cv2:
            assert input_size is not None
            self.input_size = input_size
        else:
            assert step is not None
            self.step = step
            self.shift = shift

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        height = np.random.choice(self.height)
        target = self.data[index, height]  # [5, 15, 14]
        if self.use_cv2:
            # note that cv2.resize accept (width, height) but the data shape is (height, width)
            input_ = np.array(
                [
                    cv2.resize(
                        target[i], self.input_size[::-1], interpolation=cv2.INTER_AREA
                    )
                    for i in range(target.shape[0])
                ]
            )
        else:
            input_ = target[
                :, self.shift[0] :: self.step[0], self.shift[1] :: self.step[1]
            ]
        return height, torch.from_numpy(input_), torch.from_numpy(target)


class Corner2CenterDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        means: np.ndarray,
        stds: np.ndarray,
        seg_num: tuple,
        force_square=False,
    ) -> None:
        super().__init__()

        with h5py.File(data_dir) as f:
            self.data = f["data"][:]
        self.height = self.data.shape[1]
        self.data -= means.reshape(1, *means.shape, 1, 1)
        self.data /= stds.reshape(1, *means.shape, 1, 1)

        if isinstance(seg_num, int):
            seg_num = (seg_num, seg_num)
        xs = []
        ys = []
        h, w = self.data.shape[-2:]
        for i in range(h):
            # the first element is (i, i), so we drop it.
            xs.extend(
                [(i, j + 1, k) for k, j in enumerate(range(i, h, seg_num[0]))][1:]
            )
        for i in range(w):
            ys.extend(
                [(i, j + 1, k) for k, j in enumerate(range(i, w, seg_num[1]))][1:]
            )
        if force_square:
            self.anchors = [
                (x, y) for x, y in itertools.product(xs, ys) if x[2] == y[2]
            ]
        else:
            self.anchors = list(itertools.product(xs, ys))

    def __len__(self):
        return len(self.anchors) * self.data.shape[0]

    def __getitem__(self, index):
        sample_idx, anchor_idx = divmod(index, len(self.anchors))
        x, y = self.anchors[anchor_idx]
        height = np.random.choice(self.height)
        data = torch.from_numpy(self.data[sample_idx, height])
        input_ = data[:, (x[0], x[1])][..., (y[0], y[1])]
        target = data[:, x[0] : x[1] : x[2], y[0] : y[1] : y[2]]
        return height, input_, target


def test_dataset1():
    means = np.load("./data/processed/mean.npy")
    stds = np.load("./data/processed/std.npy")
    data_dir = "./data/processed/data.h5"
    dataset = SuperResoNC(data_dir, means, stds, use_cv2=True, input_size=(10, 8))
    rprint(len(dataset))
    rprint(dataset[1234][1].shape)


def test_dataset2():
    means = np.load("./data/processed/mean.npy")
    stds = np.load("./data/processed/std.npy")
    data_dir = "./data/processed/data.h5"
    dataset = Corner2CenterDataset(data_dir, means, stds, seg_num=2, force_square=True)
    rprint(len(dataset))
    rprint(dataset[1234][1].shape)  # (5, 2, 2)
    rprint(dataset[1233][2].shape)  # (5, seg_num + 1, seg_num + 1)


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
    test_dataset2()
