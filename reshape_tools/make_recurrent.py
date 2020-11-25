import numpy as np
import pandas as pd
from warnings import warn
from nptyping import NDArray
from typing import Any, Optional
from logging import Logger
from reshape_logging.loggers import CustomLogger


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
    # order_ix = order_ixs[0]

    # Iterate over rows and reshape.
    # if logger is not None:
        # logger.info(f"Processing a dataframe of size {df.shape}")
    for ix, row in df.iterrows():

        if ix + time_window > df.shape[0]:
            break  # Breaks if no more rows are available.

        # Grab a chunk of rows equal to teh size of the time_window.
        sub_df = df.iloc[ix : ix + time_window, :]

        if partition_by is not None:
            if not (
                sub_df[partition_by].values[0] == sub_df[partition_by].values
            ).all():
                continue  # If we're using partition_by, then skip if any value of the partition column is different.
        if drop_order_by:
            sub_df = sub_df.drop(order_by, axis=1)
        if drop_partition_by and partition_by is not None:
            sub_df = sub_df.drop(partition_by, axis=1)
        arr = sub_df.values
        # if logger is not None:
            # logger.debug(f"Iteration at row {ix} of {df.shape[0]} produced an array of size {arr.shape}")
        arrs.append(arr.reshape(1, -1, arr.shape[1]))

    if len(arrs) == 0:
        warn(f"in make_recurrent(), partition key {partition_by} yielded 0 arrays")
        return None
    return np.concatenate(arrs, axis=0)
