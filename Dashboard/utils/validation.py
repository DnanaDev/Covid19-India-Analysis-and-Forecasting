"""
Function used for Parser Data validation.
"""
import numpy as np
import pandas as pd


def validate_monotonicity(series):
    """Check and correct monotonicity of cumulative Testing samples series.
    The source often has errors where incorrect numbers are entered. Causes the testing graph to go crazy."""
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
