import math

import numpy as np
import pandas as pd
from scipy import io


pred_data_dir = "./data/pred/submit_dummy0.mat"
val_data_dir = "./data/raw/processed/"


# def calc_distance(lon1, lon2, lat1, lat2):
#     """https://www.kite.com/python/answers/how-to-find-the-distance-between-two-lat-long-coordinates-in-python
#     Calculate distance from geographical coordinates.
#     """
#     R = 6373.0  # radius of the Earth
#     lon1 = math.radians(lon1)
#     lon2 = math.radians(lon2)
#     lat1 = math.radians(lat1)
#     lat2 = math.radians(lat2)
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = (
#         math.sin(dlat / 2) ** 2
#         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
#     )  # Haversine formula
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     distance = R * c
#     return distance


# def calc_distance(lon1, lon2, lat1, lat2):
#     delta_lon = (lon1 - lon2) / 0.009168333333333105
#     delta_lat = (lat1 - lat2) / 0.008425000000000031
#     return delta_lon ** 2 + delta_lat ** 2
DELTA_LON = 0.009168333333333105
DELTA_LAT = 0.008425000000000031


def val(pred_data_dir, val_info_dir, val_data_dir, date="2019-06-30", on_time=True):
    submit = io.loadmat(pred_data_dir)
    assert sorted(submit.keys()) == sorted(
        [
            "__globals__",
            "__header__",
            "__version__",
            "Lon",  # [150, 140]
            "Lat",
            "F3",  # [150, 140, 1, 24] / [150, 140, 1, 24 * 4]
            "F8",
            "F10",
            "F14",
            "F15",
            "F17",
            "F19",
            "F20",
            "F21",
            "F22",
            "F23",
            "F25",
        ]
    ), sorted(submit.keys())
    station_info = pd.read_csv(val_info_dir)
    station_data = pd.read_csv(val_data_dir).query(f'date == "{date}"')
    station_data["mask"] = station_data["mask"].astype(bool)
    assert len(station_data) == 12 * 24 * 4
    if on_time:
        station_data = station_data[station_data.time.str.contains(":00")]
        assert len(station_data) == 12 * 24

    maes = []
    for _, i in station_info.iterrows():
        distances = np.power((submit["Lon"] - i.lon) / DELTA_LON, 2) + np.power(
            (submit["Lat"] - i.lat) / DELTA_LAT, 2
        )
        nearest_coor_lon, nearest_coor_lat = divmod(distances.argmin(), 140)
        pred = submit[i.station_id][nearest_coor_lon, nearest_coor_lat, 0]
        target = station_data.query(f'station_id == "{i.station_id}"')
        maes.extend(
            np.abs(pred[~target["mask"]] - target.wind_speed[~target["mask"]]).tolist()
        )
    return 1 / (1 + sum(maes) / len(maes))


def test_example():
    rprint(
        val(
            "./data/submit/submit_median_gt.mat",
            "./data/raw/processed/train_station.csv",
            "./data/processed/train.csv",
        )
    )


if __name__ == "__main__":
    from rich import print as rprint
    from rich.traceback import install

    install()
    test_example()
