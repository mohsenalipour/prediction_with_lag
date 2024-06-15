
def make_lag(df, target_col, n_lags):
    for i in range(1, n_lags+1):
        df[f'lag_{target_col}_{i}'] = df[target_col].shift(i)

    return df


def make_lag_shift(df, target_col, n_lags, row):
    df_new = df.copy()
    index = df_new.index[row]
    for i in range(1, n_lags+1):
        df_new.loc[index, f'lag_{target_col}_{i}'] = df_new.loc[index - i, 'target']
    return df_new


def make_target(df, target_col, shift):
    df['target'] = df[target_col].shift(shift)
    df.drop(columns=[target_col], inplace=True)
    return df
