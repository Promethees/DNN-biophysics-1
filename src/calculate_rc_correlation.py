import numpy as np
import tensorflow as tf
import os
from model import build_model
from utils import load_data, preprocess_data, env
import matplotlib.pyplot as plt
import seaborn as sns

# Set environment (vacuum or water)

# Load and preprocess data (same dataset for all predictions)
X, y = load_data(f"../data/{env}_data.csv")
X_scaled, scaler = preprocess_data(X)

# Number of training sets
num_sets = 10

# Store predictions for each model's RC
predictions = []

# Function to read hyperparameters from best_hps_{env}.txt
def read_best_hps(file_path):
    hps = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if key.startswith('n_nodes_') or key == 'n_layers':
                hps[key] = int(value)
            else:  # l2_lambda_
                hps[key] = float(value)
    return hps

# Iterate over the 10 training sets
for i in range(1, num_sets + 1):
    # Load hyperparameters
    hp_file = f"../results/{env}/hyperparameters/set_{i}/best_hps_{env}.txt"
    if not os.path.exists(hp_file):
        print(f"Warning: {hp_file} not found, skipping set {i}")
        continue
    hps = read_best_hps(hp_file)
    
    # Build model
    n_layers = hps['n_layers']
    n_nodes = [hps[f'n_nodes_{j}'] for j in range(n_layers)]
    l2_lambda = [hps[f'l2_lambda_{j}'] for j in range(n_layers)]
    input_dim = 90 if env == "vacuum" else 134
    model = build_model(n_layers, n_nodes, l2_lambda, input_dim)
    
    # Load model weights (assumes weights are saved during tuning)
    model_path = f"../results/{env}/hyperparameters/set_{i}/best_model"
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        print(f"Warning: Model weights not found at {model_path}, skipping set {i}")
        continue
    
    # Predict RC (q_i) for all configurations
    q_i = model.predict(X_scaled, batch_size=32, verbose=0).flatten()
    predictions.append(q_i)

# Convert predictions to a NumPy array (shape: [num_configs, num_sets])
predictions = np.array(predictions).T  # Shape: [num_configs, num_sets]

# Calculate correlation matrix
corr_matrix = np.corrcoef(predictions, rowvar=False)

# Save correlation matrix to file
output_dir = f"../results/{env}/correlations"
os.makedirs(output_dir, exist_ok=True)
np.savetxt(os.path.join(output_dir, f"rc_correlation_matrix_{env}.txt"), corr_matrix, fmt='%.6f')

# Visualize correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.3f')
plt.title(f"Correlation Matrix of RCs ({env.capitalize()})")
plt.xlabel("Training Set Index")
plt.ylabel("Training Set Index")
plt.savefig(os.path.join(output_dir, f"rc_correlation_matrix_{env}.png"))
plt.close()

print(f"Correlation matrix saved to {output_dir}/rc_correlation_matrix_{env}.txt")
print(f"Heatmap saved to {output_dir}/rc_correlation_matrix_{env}.png")