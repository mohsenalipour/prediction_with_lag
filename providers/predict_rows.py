import pandas as pd


def predict_every_rows(df, n_predict, n_lags, target, model, predict_one_function):
    feature_cols = [col for col in df.columns if col != 'target']
    new_dict = {key: [] for key in feature_cols}
    new_dict['index'] = []
    new_dict['target'] = []

    m = 1
    last_row = {}
    for index, row in df.iterrows():
        new_dict['index'].append(m)
        targets_shift = []
        if m <= n_lags:
            empty_cols_number = m - 1

            if empty_cols_number != 0:
                # fill blank cols
                for i in range(1, empty_cols_number+1):
                    new_dict[f'lag_{target}_{i}'] = targets_shift[-1]

            for col in feature_cols:
                new_dict[col].append(row[col])
            predict_target = predict_one_function(model, row[:-1])
            new_dict['target'].append(predict_target)
            targets_shift.append(predict_target)

        m += 1

    df_final = pd.DataFrame(new_dict)

    return df_final
