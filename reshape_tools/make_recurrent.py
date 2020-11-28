import numpy as np
import pandas as pd
from nptyping import NDArray
from typing import Any, Optional
from warnings import warn


GDC_TRANSFORMATIONS = {
    "m": lambda x: x.dt.to_period("M"),
    "month": lambda x: x.dt.to_period("M"),
    "d": lambda x: x.dt.to_period("D"),
    "day": lambda x: x.dt.to_period("D"),
    "y": lambda x: x.dt.to_period("Y"),
    "year": lambda x: x.dt.to_period("Y"),
}


def get_date_component(dates: pd.Series, component: str) -> pd.Series:
    """Returns dates aggregated at day, month, or year level.

    Args:
        dates (pd.Series): A series of dates to transform.
        component (str): The component you wish to pull.

    Returns:
        A series of the components.
    """
    gdc_transformation = GDC_TRANSFORMATIONS[component.lower()]
    periods = gdc_transformation(dates)
    return periods.astype(str)


def make_recurrent(
    df: pd.DataFrame,
    time_window: int,
    order_by: str,
    partition_by: Optional[str] = None,
    drop_order_by: bool = True,
    drop_partition_by: bool = True,
    ascending: bool = True,
    verbose: bool = False,
    strict: bool = False,
) -> Optional[NDArray[(Any, Any, Any)]]:
    """Converts a 2-dimensional dataframe into a 3-dimensional recurrent.

    Args:
        df (pd.DataFrame): The data you wish to reshape.
        time_window (int): The number of observations you want in the time dimension of the array.
        order_by (str): The column you wish to sort your observations by.
        partition_by (str): The (optional) column you wish to partition by (e.g. if you have observations
            you do not want in the same time series, then partition by the column that differentiates them).
        drop_order_by (bool): Whether to drop the order_by column. Defaults to True.
        drop_partition_by (bool): Whether to drop the partition_by column. Defaults to True.
        ascending (bool): Ascending argument passed onto sort_values method of DataFrame.

    Returns:
        np.ndarray: A 3-dimensional numpy array. Observations are dimension 0, timesteps are dimension 1,
            different variables are dimension 2.
    """

    # Set up some variables we'll use later.
    arrs = []
    if df.shape[0] == 0:
        raise ValueError("An empty dataframe was passed to make_recurrent()")
    df = df.sort_values(order_by, ascending=ascending)
    assert df.shape[0] > 0
    df_colnames = ", ".join([col for col in df.columns.values])
    order_ixs = [ix for ix, val in enumerate(df.columns.values) if order_by == val]

    # Check if the order_by column is actually in the df columns.
    if len(order_ixs) == 0:
        raise ValueError(
            f"order_by column {order_by} was not found in colnames {df_colnames}."
        )
    # If we're using partition_by, check if it is actually in the df columns.
    if partition_by is not None:
        if partition_by not in df.columns.values:
            raise ValueError(
                f"partition_by column {partition_by} was not None, but was not found in colnames {df_colnames}"
            )

    for ix, row in df.iterrows():
        if ix + time_window > df.shape[0]:
            if verbose:
                print(f"No remaining full time windows in dataframe of shape {df.shape} because ix of {ix} + time_window of {time_window} exceeded its shape. Ending")
            break  # Breaks if no more rows are available.

        # Grab a chunk of rows equal to the size of the time_window.
        sub_df = df.iloc[ix : ix + time_window, :]
        assert sub_df.shape[0] > 0

        if partition_by is not None:
            partition_by_values = sub_df[partition_by].values
            if not (
                sub_df[partition_by].values[0] == partition_by_values
            ).all():
                if verbose:
                    print(f"Partition condition failed. Skipping to next window. Unique partition_by values in sub_df included: {', '.join(np.unique(partition_by_values))}")
                continue  # If we're using partition_by, then skip if any value of the partition column is different.
        if drop_order_by:
            sub_df = sub_df.drop(order_by, axis=1)
        if drop_partition_by and partition_by is not None:
            sub_df = sub_df.drop(partition_by, axis=1)
        if verbose:
            print("Adding row to array.")
        arr = sub_df.values
        arrs.append(arr.reshape(1, -1, arr.shape[1]))

    if len(arrs) == 0:
        if strict:
            raise Exception(f"in make_recurrent(), partition key {partition_by} yielded 0 arrays")
        else:
            warn(f"in make_recurrent(), partition key {partition_by} yielded 0 arrays")
            return None
    return np.concatenate(arrs, axis=0)
