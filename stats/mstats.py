import numpy as np
import pandas as pd

def avgMedDatetime(dtSeries, indicateNullValues=False):
    """
    INPUT:
        dtSeries		pd.Series, datetime values
        indicateNullValues	bool, print counts of null values, original
                                series length, length after dropping nulls,
                                default: False

    Given a series of datetime values, obtains the mean and median values.
    (Null values are dropped prior to computations.)
    """

    dtMin = dtSeries.min()
    dtSeriesFinite = list(filter(lambda dt: not pd.isnull(dt), dtSeries))
    Δs = [(dt - dtMin) for dt in dtSeriesFinite]
    if indicateNullValues:
        print(f"nulls: {np.count_nonzero(pd.isnull(dtSeries))}, "
              f"original: {len(dtSeries)}, "
              f"final: {len(dtSeriesFinite)}")

    dtAvg = dtMin + pd.Series(Δs).mean()
    dtMed = dtMin + pd.Series(Δs).median()

    return dtAvg, dtMed
