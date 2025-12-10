# utils.py

import os
import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def load_data(path="uber.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def prepare_features(df, target_col='fare_amount'):
    df = df.copy()

    if target_col in df.columns:
        df = df[df[target_col].notna()]

    # Detect datetime columns
    dt_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if len(dt_cols) > 0:
        dt = dt_cols[0]
        df[dt] = pd.to_datetime(df[dt], errors='coerce')
        df['hour'] = df[dt].dt.hour
        df['dayofweek'] = df[dt].dt.dayofweek

    # Compute distance
    coords = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    if all(c in df.columns for c in coords):
        df['distance_km'] = haversine_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )

    feature_cols = []
    for col in ['distance_km', 'hour', 'dayofweek', 'passenger_count']:
        if col in df.columns:
            feature_cols.append(col)

    df = df.dropna(subset=feature_cols + ([target_col] if target_col in df.columns else []))

    X = df[feature_cols]
    y = df[target_col] if target_col in df.columns else None
    return X, y, feature_cols
