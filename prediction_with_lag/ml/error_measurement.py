from sklearn.metrics import mean_squared_error


def error_measurement(y_pred, y_test, metric='mean_squared_error'):
    if metric == 'mean_squared_error':
        error = mean_squared_error(y_true=y_test, y_pred=y_pred)

    return error
