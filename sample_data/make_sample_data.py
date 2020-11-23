import pandas as pd
from datetime import datetime


def sample_data1() -> pd.DataFrame:
    times = [
        datetime(year=2020, month=11, day=1),
        datetime(year=2020, month=11, day=2),
        datetime(year=2020, month=11, day=3),
        datetime(year=2020, month=11, day=4),
        datetime(year=2020, month=11, day=5),
        datetime(year=2020, month=12, day=6),
        datetime(year=2020, month=12, day=7),
        datetime(year=2020, month=12, day=8),
        datetime(year=2020, month=12, day=9),
        datetime(year=2020, month=12, day=10),
    ]

    val1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    val2 = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    return pd.DataFrame({"times": times, "val1": val1, "val2": val2})


def sample_data2() -> pd.DataFrame:
    times = [
        datetime(year=2020, month=11, day=1, hour=8),
        datetime(year=2020, month=11, day=1, hour=9),
        datetime(year=2020, month=11, day=1, hour=10),
        datetime(year=2020, month=11, day=1, hour=11),
        datetime(year=2020, month=11, day=1, hour=12),
        datetime(year=2020, month=11, day=2, hour=8),
        datetime(year=2020, month=11, day=2, hour=9),
        datetime(year=2020, month=11, day=2, hour=10),
        datetime(year=2020, month=11, day=2, hour=11),
        datetime(year=2020, month=11, day=2, hour=12),
    ]

    val1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    val2 = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    return pd.DataFrame({"times": times, "val1": val1, "val2": val2})
