import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df):
    """Remove nulls and handle basic cleaning."""
    df = df.dropna()
    return df

def encode_labels(df, target_col):
    """Label-encode the target column if it's categorical."""
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        return df, le
    return df, None

def scale_features(X):
    """Standard scale the feature set."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def prepare_features(df, target_col):
    """Splits dataframe into X and y, applies label encoding and scaling."""
    df = clean_data(df)
    df, encoder = encode_labels(df, target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_scaled, scaler = scale_features(X)
    return X_scaled, y, encoder, scaler
