import tensorflow as tf
from model import build_model, predict_p_B
from utils import load_data, preprocess_data
import numpy as np
from sklearn.metrics import mean_squared_error

# Load best hyperparameters
best_hps = {}
with open('../results/hyperparameters/best_hps.txt', 'r') as f:
    lines = f.readlines()
    best_hps['n_layers'] = int(lines[0].split(': ')[1])
    for i in range(best_hps['n_layers']):
        best_hps[f'n_nodes_{i}'] = int(lines[2*i+1].split(': ')[1])
        best_hps[f'l2_lambda_{i}'] = float(lines[2*i+2].split(': ')[1])

# Load and preprocess data
X, y = load_data('../data/vacuum_data.csv')  # Or water_data.csv
X_scaled, scaler = preprocess_data(X)

# Set input_dim based on dataset
input_dim = 90 if 'vacuum' in data_file else 134

# Split data (50% train, 10% val, 40% test)
n = len(X)
n_train = int(n * 0.5)
n_val = int(n * 0.1)
indices = np.random.permutation(n)
train_idx, val_idx, test_idx = indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
X_train, y_train = X_scaled[train_idx], y[train_idx]
X_val, y_val = X_scaled[val_idx], y[val_idx]
X_test, y_test = X_scaled[test_idx], y[test_idx]

# Build and train model
model = build_model(best_hps['n_layers'],
                    [best_hps[f'n_nodes_{i}'] for i in range(best_hps['n_layers'])],
                    [best_hps[f'l2_lambda_{i}'] for i in range(best_hps['n_layers'])], 
                    input_dim)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), callbacks=[stop_early])

# Evaluate and save model
y_pred = predict_p_B(model, X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")
model.save('../results/models/final_model.h5')