import numpy as np

from reshape_tools.make_recurrent import get_date_component
from sample_data.make_sample_data import sample_data1, sample_data2


def test_gdc1():
    df = sample_data1()
    d = get_date_component(df["times"], "month")

    assert len(d) == df.shape[0]
    unique_months = np.sort(np.unique(d))
    assert len(unique_months) == 2
    assert unique_months[0] == "2020-11"
    assert unique_months[1] == "2020-12"


def test_gdc2():
    df = sample_data1()
    d = get_date_component(df["times"], "day")

    assert len(d) == df.shape[0]
    unique_days = np.sort(np.unique(d))
    assert len(unique_days) == 10
    assert unique_days[0] == "2020-11-01"
    assert unique_days[1] == "2020-11-02"


def test_gdc3():
    df = sample_data1()
    d = get_date_component(df["times"], "year")

    assert len(d) == df.shape[0]
    unique_years = np.sort(np.unique(d))
    assert len(unique_years) == 1
    assert unique_years[0] == "2020"
