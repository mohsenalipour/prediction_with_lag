import numpy as np
import pandas as pd

from providers.data_prepare import make_n_days


def predict_with_model(model, X_test):
    val = np.array(X_test)
    y_pred = model.predict(val)
    return y_pred


def predict_with_model_one(model, val):
    val = np.array(val).reshape(1, -1)
    y_pred = model.predict(val)
    return y_pred[0]


def predict_close_with_predicted_returns(base_df, prediction_df, target_col, n_predict):
    df = base_df.loc[:, ['Date', 'Close', target_col]]
    df.set_index('Date', inplace=True, drop=True)
    dates = make_n_days(start_date=df.index.max(), n=n_predict+1)[1:]
    pred_df = prediction_df.loc[:, ['target']]
    pred_df.rename(columns={'target': target_col}, inplace=True)
    pred_df.index = dates
    final_df = pd.concat([df, pred_df], axis=0)

    return final_df


def calculate_close_with_predicted_returns(df, target_col):
    df['Close_1'] = df['Close'].shift(1)

    close_shift = 0
    new_dict = {'index': [],
                'close': [],
                'close_1': [],
                target_col: [],
                'new_close': []}
    for index, row in df.iterrows():
        new_dict['index'].append(index)
        new_dict[target_col].append(row[target_col])
        if np.isnan(row['Close']):
            if not np.isnan(row['Close_1']):
                new_dict['close'].append(np.nan)
                new_dict['close_1'].append(row['Close_1'])
                res = np.round(np.exp(row[target_col]) * row['Close_1'], 0)
                new_dict['new_close'].append(res)
                close_shift = res
            else:
                new_dict['close'].append(np.nan)
                new_dict['close_1'].append(close_shift)
                res = np.round(np.exp(row[target_col]) * close_shift, 0)
                new_dict['new_close'].append(res)
                close_shift = res
        else:
            new_dict['close'].append(row['Close'])
            new_dict['close_1'].append(row['Close_1'])
            res = np.round(np.exp(row[target_col]) * row['Close_1'], 0)
            new_dict['new_close'].append(res)
            close_shift = row['Close']

    df_top = pd.DataFrame(new_dict, columns=new_dict.keys())
    df_top.set_index('index', inplace=True, drop=True)
    df_top['Close'] = df_top['new_close']
    df_top = df_top.loc[:, ['Close', target_col]]

    return df_top
