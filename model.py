# model.py

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os


def train_and_evaluate(X, y, model_name='RandomForest', random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    mae = mean_absolute_error(y_test, preds)

    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, preds))

    r2 = r2_score(y_test, preds)

    return {
        'model': model,
        'scaler': scaler,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'X_test': X_test,
        'y_test': y_test,
        'preds': preds
    }


def save_model(model, scaler, model_path, scaler_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def load_model(model_path, scaler_path):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)
