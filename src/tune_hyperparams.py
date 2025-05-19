import keras_tuner as kt
import tensorflow as tf
from model import build_model
from utils import load_data, preprocess_data, split_data
import numpy as np

def model_builder(hp):
    n_layers = hp.Int('n_layers', 2, 5)
    n_nodes = [hp.Int(f'n_nodes_{i}', 100, 5000, step=100) for i in range(n_layers)]
    l2_lambda = [hp.Float(f'l2_lambda_{i}', 0.0001, 0.1, sampling='log') for i in range(n_layers)]
    # Determine input_dim based on dataset (hardcoded for now, can be made dynamic)
    input_dim = 90  # Default for vacuum
    model = build_model(n_layers, n_nodes, l2_lambda, input_dim)
    return model

# Load and preprocess data
X, y = load_data('../data/vacuum_data.csv')  # Or water_data.csv
X_scaled, scaler = preprocess_data(X)
X_train, y_train, X_val, y_val = split_data(X_scaled, y)

# Set up tuner
tuner = kt.BayesianOptimization(
    model_builder,
    objective='val_loss',
    max_trials=150,
    directory='../results/hyperparameters',
    project_name='alanine_dnn'
)

# Early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Search for best hyperparameters
tuner.search(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), callbacks=[stop_early])

# Save best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
with open('../results/hyperparameters/best_hps.txt', 'w') as f:
    f.write(f"n_layers: {best_hps.get('n_layers')}\n")
    for i in range(best_hps.get('n_layers')):
        f.write(f"n_nodes_{i}: {best_hps.get(f'n_nodes_{i}')}\n")
        f.write(f"l2_lambda_{i}: {best_hps.get(f'l2_lambda_{i}')}\n")