import xgboost as xgb


# XGBoost
def xgboost_model_fitter(X_train, y_train, parameters=None):
    if parameters is None:
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.08,
            subsample=0.75,
            colsample_bytree=1,
            max_depth=7,
            gamma=0,
        )
    else:
        model = xgb.XGBRegressor(**parameters)
    model.fit(X_train, y_train)
    return model
