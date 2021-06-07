import os
import numpy as np
import pandas as pd
import h5py
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from rich import print as rprint


def inspect_data_structure(raw_dir):
    """Inspect the data structure of given .nc file with the package `netCDF4`
    """
    nc_example = raw_dir + 'nc_of_201904-201906/201904/201904010000ft.nc'
    dataset = nc.Dataset(nc_example)
    rprint(dataset)
    rprint(dir(dataset))
    rprint()

    rprint('==== dimensions ====')
    for dim in dataset.dimensions.values():
        rprint(dim)
        rprint()

    rprint('==== variables ====')
    for var in dataset.variables.values():
        rprint(var.name)
        rprint(var)
        rprint()

    rprint('==== time ====')
    rprint(np.array(dataset['Times']).astype(str))
    rprint()

    rprint('==== coordinates ====')
    rprint(np.array(dataset['Lon'][:, 0]))
    rprint(np.array(dataset['Lat'][0]))


def collect_data(raw_dir, processed_dir, img_dir):
    """Collect all raw data together, fill nan value, and store into a hdf5 file.
    """
    nc_dir = raw_dir + 'nc_of_201904-201906/'
    nc_example = nc_dir + '201904/201904010000ft.nc'
    example_dataset = nc.Dataset(nc_example)

    # ---- collect data into a dictionary ----
    data_dict = {}
    for var in example_dataset.variables.keys():
        rprint(var)
        data_dict[var] = []
    for month in sorted(os.listdir(nc_dir)):
        for nc_file in sorted(os.listdir(os.path.join(nc_dir, month))):
            dataset = nc.Dataset(os.path.join(nc_dir, month, nc_file))
            assert dataset.variables.keys() == data_dict.keys()
            for key in dataset.variables.keys():
                data_dict[key].append(np.array(dataset[key]))
    assert len(data_dict['tsig']) == (30 + 31 + 30) * 24 * 4
    
    for k, v in data_dict.items():
        data_dict[k] = np.stack(v)
        rprint(data_dict[k].shape)
    
    # ---- simple eda ----
    # Time series of different features at the top left point across all heights.
    # there is nan value except for the first height.
    f, axs = plt.subplots(5, 1, figsize=(16, 8 * 5))
    for i, name, ylim in enumerate(zip(['psig', 'rhsig', 'tsig', 'tsig', 'usig'], [[850, 1050], [50, 110], [10, 30], None, None])):
        pd.DataFrame(data_dict['rhsig'][:800, :, 0, 0]).plot(kind='line', ax=axs[i], ylim=ylim)
    plt.savefig(img_dir + 'before_fillna.jpg')

    # Time series comparing features across 15 differnt longitude at the zeroth height and smallest latitude.
    _, axs = plt.subplots(5, 1, figsize=(16, 8 * 5))
    for i, name in enumerate(['tsig', 'rhsig', 'usig', 'vsig', 'psig']):
        pd.DataFrame(data_dict[name][:800, 0, :, -1]).iloc[:, :].plot(kind='line', ax=axs[i])
    plt.savefig(img_dir + 'across_lon.jpg')

    # A panoramic view of time series of usig and vsig
    _, (ax1, ax2) = plt.subplots(3, 1, figsize=(16, 8 * 3))
    pd.Series(data_dict['usig'][:, 0, 0, 0], name='usig').plot(kind='line', ax=ax1)
    pd.Series(data_dict['vsig'][:, 0, 0, 0], name='vsig').plot(kind='line', ax=ax2)
    wind_speed = np.sqrt(np.power(data_dict['usig'][:, 0, 0, 0], 2) + np.power(data_dict['vsig'][:, 0, 0, 0], 2))
    pd.Series(wind_speed, name='wind speed').plot(kind='line', ax=ax2)
    plt.savefig(img_dir + 'whole_time_series.jpg')

    # ---- fillna... ----
    for k in ('psig', 'rhsig', 'tsig'):
        data_dict[k] = fillna(data_dict[k])

    # ---- and nan value disappears ----
    _, axs = plt.subplots(5, 1, figsize=(16, 8 * 5))
    for i, name, ylim in enumerate(['psig', 'rhsig', 'tsig', 'tsig', 'usig']):
        pd.DataFrame(data_dict['rhsig'][:800, :, 0, 0]).plot(kind='line', ax=axs[i], ylim=ylim)
    plt.savefig(img_dir + 'after_fillna.jpg')

    # ---- store into hdf5 ----
    with h5py.File(processed_dir, 'w') as f:
        data = np.stack([data_dict[k] for k in ('psig', 'rhsig', 'tsig', 'usig', 'vsig')], axis=2)
        mean = data.mean(axis=(0, 3, 4))
        std = data.std(axis=(0, 3, 4))
        f['data'] = data
        f['altitude'] = data_dict['height'][0]
        f['longitude'] = data_dict['Lon'][:, 0]
        f['latitude'] = data_dict['Lat'][0]
        f['mean'] = mean
        f['std'] = std
    
    # ---- plot dist ----
    _, axs = plt.subplots(12, 5, figsize=(20, 24))
    for i in tqdm(range(12)):
        for j in range(5):
            pd.DataFrame((data[:, i, j, :, :].flatten() - mean[i, j]) / std[i, j]).plot(kind='hist', bins=30, ax=axs[i, j])
    plt.savefig('./imgs/dist.png')


def fillna(data):
    assert data.shape == (8736, 12, 15, 14)
    data = data.transpose(3, 2, 0, 1).reshape(-1, 12)
    data = np.where(data == data.min(), np.nan, data)
    data = pd.DataFrame(data).fillna(method='ffill').values.reshape(14, 15, -1, 12).transpose(2, 3, 1, 0)
    return data


def process_tsv():
    """Script to process `train.tsv`, `train_station.tsv` and `test_station.tsv`
    """


if __name__ == '__main__':
    raw_dir = './data/raw/'
    processed_dir = './data/processed/data.h5'
    img_dir = './imgs/'
    inspect_data_structure()
