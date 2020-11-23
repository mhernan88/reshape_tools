import numpy as np
import pandas as pd
from reshape_tools.make_recurrent import make_recurrent
from sample_data.make_sample_data import sample_data1, sample_data2
from nptyping import NDArray
from typing import Any, Optional


def check_results(
    output: NDArray[(Any, Any, Any)],
    data_input: pd.DataFrame,
    n_recurrent_samples: int,
    partition_by: Optional[str] = None,
):

    if partition_by is not None:
        n_unique = len(np.unique(data_input[partition_by]))
        assert output.shape[0] == data_input.shape[0] - n_unique * (
            n_recurrent_samples - 1
        )
        assert output.shape[1] == n_recurrent_samples
        assert output.shape[2] == data_input.shape[1] - 2
    else:
        assert output.shape[0] == data_input.shape[0] - n_recurrent_samples + 1
        assert output.shape[1] == n_recurrent_samples
        assert output.shape[2] == data_input.shape[1] - 1


def test_make_recurrent_no_partitioning1():
    ORDER_BY = "times"
    N_RECURRENT_SAMPLES = 3

    df = sample_data1()
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES)


def test_make_recurrent_no_partitioning2():
    ORDER_BY = "times"
    N_RECURRENT_SAMPLES = 5

    df = sample_data1()
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES)


def test_make_recurrent_no_partitioning3():
    ORDER_BY = "times"
    N_RECURRENT_SAMPLES = 3

    df = sample_data2()
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES)


def test_make_recurrent_no_partitioning4():
    ORDER_BY = "times"
    N_RECURRENT_SAMPLES = 5

    df = sample_data2()
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES)


def test_make_recurrent_partitioning1():
    ORDER_BY = "times"
    PARTITION_BY = "month"
    N_RECURRENT_SAMPLES = 3

    df = sample_data1()
    df["month"] = pd.to_datetime(df["times"]).dt.to_period("M")
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY, PARTITION_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES, PARTITION_BY)


def test_make_recurrent_partitioning2():
    ORDER_BY = "times"
    PARTITION_BY = "month"
    N_RECURRENT_SAMPLES = 4

    df = sample_data1()
    df[PARTITION_BY] = pd.to_datetime(df["times"]).dt.to_period("M")
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY, PARTITION_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES, PARTITION_BY)


def test_make_recurrent_partitioning3():
    ORDER_BY = "times"
    PARTITION_BY = "month"
    N_RECURRENT_SAMPLES = 5

    df = sample_data1()
    df[PARTITION_BY] = pd.to_datetime(df["times"]).dt.to_period("M")
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY, PARTITION_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES, PARTITION_BY)


def test_make_recurrent_partitioning4():
    ORDER_BY = "times"
    PARTITION_BY = "day"
    N_RECURRENT_SAMPLES = 3

    df = sample_data2()
    df[PARTITION_BY] = pd.to_datetime(df["times"]).dt.to_period("D")
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY, PARTITION_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES, PARTITION_BY)


def test_make_recurrent_partitioning5():
    ORDER_BY = "times"
    PARTITION_BY = "day"
    N_RECURRENT_SAMPLES = 4

    df = sample_data2()
    df[PARTITION_BY] = pd.to_datetime(df["times"]).dt.to_period("D")
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY, PARTITION_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES, PARTITION_BY)


def test_make_recurrent_partitioning6():
    ORDER_BY = "times"
    PARTITION_BY = "day"
    N_RECURRENT_SAMPLES = 5

    df = sample_data2()
    df[PARTITION_BY] = pd.to_datetime(df["times"]).dt.to_period("D")
    arr = make_recurrent(df, N_RECURRENT_SAMPLES, ORDER_BY, PARTITION_BY)
    check_results(arr, df, N_RECURRENT_SAMPLES, PARTITION_BY)
