import pandas as pd


def predict_every_rows_for_lags(df, n_lags, model, predict_one_function, target_col):
    df_new = df.copy()
    for i in range(n_lags):
        x_features = df_new.iloc[i, :-1].values
        target = predict_one_function(model, x_features)
        m = i + 1
        n = 1
        while m <= n_lags and n <= n_lags:
            row = m+1
            col = n
            # print([row, col])
            df_new.iloc[i, -1] = target
            if row <= n_lags:
                df_new.loc[df_new.index[row-1], [f'lag_{target_col}_{col}']] = target
            m += 1
            n += 1
        # print(target)

    return df_new


def predict_every_row_after_lags(df, n_lags, model, predict_one_function, target_col):
    for i in range(n_lags, df.shape[0]):
        pass
