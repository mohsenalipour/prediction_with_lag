import numpy as np


def predict_with_model(model, X_test):
    val = np.array(X_test)
    y_pred = model.predict(val)
    return y_pred


def predict_with_model_one(model, val):
    val = np.array(val).reshape(1, -1)
    y_pred = model.predict(val)
    return y_pred[0]


def predict_with_model_day_by_day(fitted_model, all_data, n_lags, base_point, n):
    pass
