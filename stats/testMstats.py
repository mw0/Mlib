from mstats import *
import pandas as pd


def testAvgMedDatetime(capsys):
    dtSeries = pd.Series(pd.date_range('2005-01-21', periods = 7,
                                       freq='1D1H'))
    # print(dtSeries)

    dtSeries[1] = pd.NaT
    dtSeries[4] = pd.NaT
    dtSeries[5] = pd.NaT
    # print(dtSeries)

    expectedDtAvg = pd.to_datetime('2005-01-23 20:45:00')
    expectedDtMed = pd.to_datetime('2005-01-23 14:30:00 ')

    dtAvg, dtMed = avgMedDatetime(dtSeries, indicateNullValues=True)
    captured = capsys.readouterr()
    print(f"captured.out: {captured.out}")

    assert dtAvg == expectedDtAvg
    assert dtMed == expectedDtMed
    assert captured.out == "nulls: 3, original: 7, final: 4\n"
