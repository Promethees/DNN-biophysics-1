import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['p_B']).values  # 90 CVs
    y = df['p_B'].values  # Committor probabilities
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def split_data(X, y, train_ratio=0.8, val_ratio=0.2):
    n = len(X)
    n_train = int(n * train_ratio)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]