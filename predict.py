from itertools import product
import numpy as np
from scipy import io
import pandas as pd
import h5py
import torch

from train import Net


def target_coor(m0, m1, n0, n1, x0, x1, target_size):
    y0 = (n0 - m0) / (m1 - m0) * (x1 - x0) + x0
    y1 = (n1 - n0) / (m1 - m0) * (x1 - x0) + y0
    # return np.linspace(y0, y1, target_size)
    return y0, y1


def test_target_coor():
    input_size = (5, 5)
    target_size = (15, 14)
    step = (3, 3)
    shift = (1, 0)
    i = 0
    m0, m1 = np.arange(target_size[i])[shift[i] :: step[i]][[0, -1]]
    n0, n1 = np.arange(target_size[i])[[0, -1]]
    x0, x1 = -1 / 3, 3 / 3
    print(target_coor(m0, m1, n0, n1, x0, x1, target_size[i]))
    # x0, x1 = 5, 10
    # print(target_coor(m0, m1, n0, n1, x0, x1, target_size[0]))
    # x0, x1 = 10, 15
    # print(target_coor(m0, m1, n0, n1, x0, x1, target_size[0]))
    # x0, x1 = 1, 14
    # print(target_coor(m0, m1, n0, n1, x0, x1, target_size[0]))


@torch.no_grad()
def validate_unet():
    checkpoint = "./lightning_logs/version_6/checkpoints/epoch=541-step=314901.ckpt"
    model = Net.load_from_checkpoint(checkpoint).eval()
    with h5py.File("./data/processed/data_aug.h5") as f:
        mean = f["mean"][:]
        std = f["std"][:]
        data = (f["data"][:] - mean[..., None, None]) / std[..., None, None]
        lon0, lon1 = np.sort(f["longitude"][:])[[0, -1]]
        lat1, lat0 = np.sort(f["latitude"][:])[[0, -1]]
        heights = f["altitude"][:]

    date = 90
    data = torch.from_numpy(data[date * 24 * 4 : (date + 1) * 24 * 4 : 4]).float()

    preds = torch.zeros(24, 12, 6, 45, 38)
    xi_idx = [(0, 5), (5, 10), (10, 15)]
    yi_idx = [(0, 5), (4, 9), (8, 13)]
    xt_idx = [(0, 15), (15, 30), (30, 45)]
    yt_idx = [(0, 14), (12, 26), (24, 38)]
    for ((xi0, xi1), (xt0, xt1)), ((yi0, yi1), (yt0, yt1)) in product(
        zip(xi_idx, xt_idx), zip(yi_idx, yt_idx)
    ):
        for height in torch.arange(data.shape[1]):
            preds[:, height, :, xt0:xt1, yt0:yt1] = model(
                data[:, height, :, xi0:xi1, yi0:yi1], height.expand(24)
            )

    data = preds
    preds = torch.zeros(24, 12, 6, 135, 110)
    xi_idx = [(i, i + 5) for i in range(0, 45, 5)]
    yi_idx = [(i, i + 5) for i in range(1, 37, 4)]
    xt_idx = [(i, i + 15) for i in range(0, 135, 15)]
    yt_idx = [(i, i + 14) for i in range(0, 108, 12)]
    for ((xi0, xi1), (xt0, xt1)), ((yi0, yi1), (yt0, yt1)) in product(
        zip(xi_idx, xt_idx), zip(yi_idx, yt_idx)
    ):
        for height in torch.arange(data.shape[1]):
            preds[:, height, :, xt0:xt1, yt0:yt1] = model(
                data[:, height, :, xi0:xi1, yi0:yi1], height.expand(24)
            )

    preds = preds * std[..., None, None] + mean[..., None, None]
    # preds = preds[:, :, -2:].pow(2).sum(dim=2).sqrt()
    preds = preds[:, :, -1]
    print(preds.shape)

    # wind_speed = (
    #     torch.from_numpy(mean)[:, -2:]
    #     .pow(2)
    #     .sum(dim=1)
    #     .sqrt()
    #     .view(1, 12, 1, 1)
    #     .expand(24, 12, 150, 140)
    #     .clone()
    # )
    wind_speed = (
        torch.from_numpy(mean)[:, -1].view(1, 12, 1, 1).expand(24, 12, 150, 140).clone()
    )
    print(wind_speed.shape)
    wind_speed[:, :, 8:-7, 15:-15] = preds
    wind_speed = wind_speed.permute(2, 3, 1, 0).numpy()
    station_info = pd.read_csv("./data/raw/processed/train_station.csv")
    submit = {}
    for station in station_info.itertuples():
        order = np.searchsorted(heights, station.altitude)
        a = (
            1
            if order == 0
            else (station.altitude - heights[order - 1])
            / (heights[order] - heights[order - 1])
        )
        pred = wind_speed[:, :, order - 1] * (1 - a) + wind_speed[:, :, order] * a
        submit[station.station_id] = pred[:, :, None]

    lon0, lon1, lat0, lat1 = (
        lon0 + (lon1 - lon0) / 14 * (-4 / 9),
        lon0 + (lon1 - lon0) / 14 * (130 / 9),
        lat0 + (lat1 - lat0) / 13 * (1 / 3),
        lat0 + (lat1 - lat0) / 13 * (112 / 9),
    )
    lon = np.concatenate(
        [
            lon0 - np.sort(np.random.rand(8))[::-1],
            np.linspace(lon0, lon1, 135),
            lon1 + np.sort(np.random.rand(7)),
        ]
    )
    lat = np.concatenate(
        [
            lat0 + np.sort(np.random.rand(15))[::-1],
            np.linspace(lat0, lat1, 110),
            lat1 - np.sort(np.random.rand(15)),
        ]
    )
    submit["Lon"] = lon.repeat(140).reshape(150, 140)
    submit["Lat"] = lat.repeat(150).reshape(140, 150).T
    io.savemat("./data/submit/submit_train.mat", submit)
    return preds


if __name__ == "__main__":
    from rich.traceback import install

    install()
    preds = validate_unet()
