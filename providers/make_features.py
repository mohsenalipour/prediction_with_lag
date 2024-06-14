
def make_lag(df, target_col, n_lags):
    for i in range(1, n_lags+1):
        df[f'lag_{target_col}_{i}'] = df[target_col].shift(i)

    return df


def make_target(df, target_col, shift):
    df['target'] = df[target_col].shift(shift)
    df.drop(columns=[target_col], inplace=True)
    return df
