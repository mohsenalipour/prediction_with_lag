import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def raw_data(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def initial_data_prep(df, day_as_feature=True, in_pred=False):
    if day_as_feature:
        df['Day'] = df.Date.astype(str).str[8:10].astype(float)
        df['Month'] = df.Date.astype(str).str[6:7].astype(float)
        df['Year'] = df.Date.astype(str).str[0:4].astype(float)
    df.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Weekday', 'simple_returns', 'Ticker'], axis=1, inplace=True)
    if not in_pred:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
    return df


def final_prepare(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def feature_scaling(data, columns_n_list=None):
    scaler = StandardScaler()
    if columns_n_list is None:
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = data.copy()
        scaled_data[:, columns_n_list] = scaler.fit_transform(data[:, columns_n_list])
    return scaled_data


def make_n_days(start_date, n, freq='D'):
    date_series = pd.date_range(start=start_date, periods=n, freq=freq)
    return date_series
