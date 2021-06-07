"""
Scripts to generate dummy submit data
"""

import h5py
import numpy as np
import pandas as pd
from scipy import io


def gen_perfect(train_station_dir, train_dir, submit_dir):
    """
    Generate submit file that perfectly fit the wind speed of 12 train stations at each
    hour of 2019-06-30, i.e. place the coordinates of these stations at the right bottom
    corner of the predicted grid, and fill in the ground truth value at these points.
    Coordinates and wind speed of other grid points are randomly generated.

    Note that longitude is ranked from smallest to largest (from west to east) while
    latitude is ranked from largest to smallest (from north to south).
    """
    df = pd.read_csv(train_station_dir, index_col=0)
    val_df = pd.read_csv(train_dir).query('date == "2019-06-30"')
    submit = {}
    submit["Lon"] = (
        np.concatenate(
            [
                df.lon.min() - np.sort(np.random.rand(150 - 12))[::-1],
                df.lon.sort_values().values,
            ]
        )
        .repeat(140)
        .reshape(150, 140)
    )
    submit["Lat"] = (
        np.concatenate(
            [
                df.lat.max() + np.sort(np.random.rand(140 - 12))[::-1],
                df.lat.sort_values(ascending=False).values,
            ]
        )
        .repeat(150)
        .reshape(140, 150)
        .T
    )
    lon_idx = df.lon.rank().astype(int) - 1
    lat_idx = df.lat.rank(ascending=False).astype(int) - 1
    for station in df.index:
        pred = np.random.rand(150, 140, 1, 24)
        pred[lon_idx.loc[station] - 12, lat_idx.loc[station] - 12, 0] = val_df.query(
            f'station_id == "{station}"'
        ).wind_speed[::4]
        submit[station] = pred
    io.savemat(submit_dir, submit)


def gen_median_gt(train_station_dir, train_dir, data_dir, submit_dir):
    """
    Generate submit file that is filled with the median of wind speed in train.csv (1.4).
    Longitude and latitude are obtained using `np.linspace` with the maximum and minimum 
    longitude/latitude in the training grid.
    """
    df = pd.read_csv(train_station_dir, index_col=0)
    median = pd.read_csv(train_dir).wind_speed.median()
    print(median)
    median = 1.0
    submit = {}
    with h5py.File(data_dir) as f:
        submit['Lon'] = np.linspace(*f['longitude'][[0, -1]], 150).repeat(140).reshape(150, 140)
        submit['Lat'] = np.linspace(*f['longitude'][[0, -1]], 140).repeat(150).reshape(140, 150).T
    for station in df.index:
        submit[station] = np.full((150, 140, 1, 24), median)
    io.savemat(submit_dir, submit)
    

def gen_median_train(train_station_dir, data_dir, submit_dir):
    """
    Generate submit file that is filled with the median of wind speed in the training 
    data. Note that the wind speed of each station may be different in this case, 
    because the median is obtained by first *linear interpolaing the wind speed at its 
    specific height*.
    Longitude and latitude are obtained in the same way as `gen_median_gt`
    """
    submit = {}
    with h5py.File(data_dir) as f:
        wind_speed = np.sqrt(np.power(f['data'][:, :, -2:], 2).sum(axis=2))
        assert wind_speed.shape == (8736, 12, 15, 14), wind_speed.shape
        heights = f['altitude'][:]
        submit['Lon'] = np.linspace(*f['longitude'][[0, -1]], 150).repeat(140).reshape(150, 140)
        submit['Lat'] = np.linspace(*f['longitude'][[0, -1]], 140).repeat(150).reshape(140, 150).T
    
    df = pd.read_csv(train_station_dir)
    for station in df.itertuples():
        order = np.searchsorted(heights, station.altitude)
        a = (
            1
            if order == 0
            else (station.altitude - heights[order - 1])
            / (heights[order] - heights[order - 1])
        )
        median = np.median(wind_speed[:, order - 1] * (1 - a) + wind_speed[:, order] * a)
        print(median)
        submit[station.station_id] = np.full((150, 140, 1, 24), median)
    io.savemat(submit_dir, submit)


if __name__ == '__main__':
    train_station_dir = './data/raw/processed/test_station.csv'
    train_dir = './data/processed/train.csv'
    data_dir = './data/processed/data.h5'
    # gen_perfect(train_station_dir, train_dir, './data/submit/submit_dummy.mat')
    gen_median_gt(train_station_dir, train_dir, data_dir, './data/submit/submit_median_gt.mat')
    # gen_median_train(train_station_dir, data_dir, './data/submit/submit_median_train.mat')