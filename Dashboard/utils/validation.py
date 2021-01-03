"""
Function used for Parser Data validation.
"""
import numpy as np
import pandas as pd


def interpolate_missing(series):
    """Fill Missing values and dates in discontinuous Time-Series by Linear interpolation.
    For Cumulative Test Series.
    Args:
        series: pd.Dataframe/pd.Series
    Returns:
        series: pd.Dataframe/pd.Series
    """
    # reindex to fill in missing dates
    series = series.reindex(index=pd.date_range(start=series.index.min(), end=series.index.max()))
    series = series.interpolate(method='linear', axis=0)

    return series


def validate_monotonicity(series):
    """
    Check and correct monotonicity of cumulative Testing samples series.
    The source often has errors where incorrect numbers are entered. Causes the testing graph to go crazy.
    Args:
        series: pd.Series

    Returns:
        series: pd.Series
    """
    # if series isn't monotonic
    if not series.is_monotonic:
        # print('Warning: Series is not monotonic, check source.')
        # save date and make series NaN
        date = series.loc[series.diff() < 0].index
        series.loc[series.diff() < 0] = np.NaN

        # If any of the incorrect dates is the upper bound, copy the last cumulative max value.
        if any(date == series.index[-1]):
            # print('copying cummax')
            series = np.maximum.accumulate(series.to_numpy())

        # Else use linear interpolation
        else:
            # save date range
            # print('interpolating')
            date_range = (date - pd.tseries.offsets.Day(1)).append((date, date + pd.tseries.offsets.Day(1)))

            # interpolate values using neighbouring values
            series.loc[date_range] = series.loc[date_range].sort_index().interpolate(method='linear', axis=0)

        return series
    else:
        # print('Series is monotonic.')
        return series
