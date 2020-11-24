import pandas as pd


def get_date_component(dates: pd.Series, component: str) -> pd.Series:
    """Returns dates aggregated at day, month, or year level.

    Args:
        dates (pd.Series): A series of dates to transform.
        component (str): The component you wish to pull.

    Returns:
        A series of the components.
    """
    if component.lower() in ("m", "month"):
        return pd.to_datetime(dates.dt.to_period("M"))
    elif component.lower() in ("d", "day"):
        return pd.to_datetime(dates.dt.to_period("D"))
    elif component.lower() in ("y", "year"):
        return pd.to_datetime(dates.dt.to_period("Y"))
    else:
        raise ValueError("Dates argument must be one of the following: M | Month | D | Day | Y | Year")
