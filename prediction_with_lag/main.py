import pandas as pd

from prediction_with_lag.providers.data_prepare import raw_data, initial_data_prep, final_prepare, make_n_days
from prediction_with_lag.providers.make_features import make_lag, make_target, make_lag_shift
from prediction_with_lag.providers.data_split import train_test_split, x_y_split
from prediction_with_lag.providers.predict_rows import predict_every_rows_for_lags
from prediction_with_lag.ml.models import xgboost_model_fitter
from prediction_with_lag.ml.prediction import predict_with_model, predict_with_model_one,\
    predict_close_with_predicted_returns, calculate_close_with_predicted_returns
from prediction_with_lag.ml.error_measurement import error_measurement


def predict_n_days(data_source, target_col, n_lags, shift, test_perc, n_predict, day_as_feature=True):
    raw = raw_data(xlsx_file=data_source)
    initial_data = raw.copy()
    data = initial_data_prep(df=initial_data, day_as_feature=day_as_feature)
    data = make_lag(df=data, target_col=target_col, n_lags=n_lags)
    data = make_target(df=data, target_col=target_col, shift=shift)
    data = final_prepare(df=data)
    n_columns = data.shape[1]
    # print(data.to_string())
    train, test = train_test_split(data=data, test_perc=test_perc)
    X_train, y_train, X_test, y_test = x_y_split(train_data=train, test_data=test,
                                                 x_columns=[i for i in range(n_columns - 1)],
                                                 y_column=-1)

    X_train_scaled = X_train
    X_test_scaled = X_test

    model = xgboost_model_fitter(X_train=X_train_scaled, y_train=y_train)
    y_pred = predict_with_model(model=model, X_test=X_test_scaled)
    # y_pred = predict_with_model_one(model=model, val=X_test_scaled[0])

    error = error_measurement(y_pred=y_pred, y_test=y_test)

    # add n days for predict from last row of data
    raw = raw.iloc[-n_lags:, :]
    last_date = raw.iloc[-1]['Date']
    predict_dates = make_n_days(start_date=last_date, n=n_predict + 1)[1:]
    new_raw = pd.DataFrame(columns=raw.columns)
    new_raw['Date'] = predict_dates
    df_for_predict = pd.concat([raw, new_raw], ignore_index=True)
    df_for_predict = initial_data_prep(df=df_for_predict, in_pred=True)
    df_for_predict = make_lag(df=df_for_predict, target_col=target_col, n_lags=n_lags)
    df_for_predict = df_for_predict.iloc[n_lags:, :]
    df_for_predict = make_target(df=df_for_predict, target_col=target_col, shift=shift)
    df_for_predict_pre = df_for_predict.iloc[:n_lags, :]
    df_final = predict_every_rows_for_lags(df=df_for_predict_pre, n_lags=n_lags, model=model,
                                           predict_one_function=predict_with_model_one, target_col=target_col)
    new_for_predict = pd.concat([df_final, df_for_predict.iloc[n_lags:, :]], ignore_index=True)
    new_df = new_for_predict.copy()
    for i in range(n_lags, len(new_for_predict)):
        new_df = make_lag_shift(df=new_df, target_col='log_returns', n_lags=n_lags, row=i)
        x_features = new_df.iloc[i, :-1]
        new_df.loc[new_df.index[i], 'target'] = predict_with_model_one(model, x_features)

    output_target = predict_close_with_predicted_returns(base_df=raw, prediction_df=new_df, target_col=target_col,
                                                         n_predict=n_predict)

    output = calculate_close_with_predicted_returns(df=output_target, target_col=target_col)

    return output, error


output, error = predict_n_days(data_source='data/dollar.xlsx',
                               target_col='log_returns',
                               n_lags=5,
                               shift=-1,
                               test_perc=0.3,
                               n_predict=100
                               )
print(error)
print(output.to_string())
output.to_excel('output/dollar.xlsx')
