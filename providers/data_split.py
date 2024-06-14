def train_test_split(data, test_perc):
    data = data.values
    n = int(len(data) * (1 - test_perc))
    train_data = data[:n]
    test_data = data[n:]
    return train_data, test_data


def x_y_split(train_data, test_data, x_columns: list, y_column: int):
    X_train = train_data[:, x_columns]
    y_train = train_data[:, y_column]
    X_test = test_data[:, x_columns]
    y_test = test_data[:, y_column]
    return X_train, y_train, X_test, y_test
