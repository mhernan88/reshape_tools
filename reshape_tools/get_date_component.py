import pandas as pd


GDC_TRANSFORMATIONS = {
    "m": lambda x: x.dt.to_period("M"),
    "month": lambda x: x.dt.to_period("M"),
    "d": lambda x: x.dt.to_period("D"),
    "day": lambda x: x.dt.to_period("D"),
    "y": lambda x: x.dt.to_period("Y"),
    "year": lambda x: x.dt.to_period("Y")
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
