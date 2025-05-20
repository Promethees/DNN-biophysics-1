import lime.lime_tabular
import shap
import pandas as pd
import numpy as np
from model import build_model  # To access the model structure
from utils import load_data, preprocess_data, env
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import os

# Load data and determine environment
data_file = f"../data/{env}_data.csv"
X, y = load_data(data_file)
X_scaled, scaler = preprocess_data(X)
input_dim = 90 if env == "vacuum" else 134

# Load the trained model
model_file = f'../results/{env}/models/final_model_{env}.h5'
model = load_model(model_file)

# Extract logits (pre-sigmoid output) as an approximation of q
# Create a new model that outputs the logits before the sigmoid activation
base_model = Model(inputs=model.input, outputs=model.layers[-2].output)  # Last layer before output
q_values = base_model.predict(X_scaled)  # Shape: (n_samples, 1)

# Split data into training, validation, and test sets (approximate splits)
n_samples = len(X_scaled)
n_train = int(n_samples * 0.5)
n_val = int(n_samples * 0.1)
indices = np.random.permutation(n_samples)
train_idx, val_idx, test_idx = indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
X_train, X_val, X_test = X_scaled[train_idx], X_scaled[val_idx], X_scaled[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
q_train, q_val, q_test = q_values[train_idx], q_values[val_idx], q_values[test_idx]

# Filter data within -0.2 < q < 0.2 for histogram (Figure S3)
def filter_q_range(q, low=-0.2, high=0.2):
    return np.logical_and(q >= low, q <= high)

mask = filter_q_range(q_values)
q_filtered = q_values[mask]
p_b_filtered = y[mask]
train_mask = mask[train_idx]
val_mask = mask[val_idx]
test_mask = mask[test_idx]

# Create plots directory if it doesn't exist
plot_path = f'../results/{env}/plots'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Figure S2: Scatter plot of q vs p_B
plt.figure(figsize=(8, 6))
scatter = plt.scatter(q_train, y_train, c='orange', label='Training', alpha=0.5)
scatter = plt.scatter(q_val, y_val, c='green', label='Validation', alpha=0.5)
scatter = plt.scatter(q_test, y_test, c='blue', label='Test', alpha=0.5)
plt.title(f'Scatter Plot of q vs p_B in {env}')
plt.xlabel('q')
plt.ylabel('p_B')
plt.legend()
plt.savefig(f'../results/{env}/plots/scatter_q_pB_{env}.png')
plt.close()

# Figure S3: Histogram of committors for -0.2 < q < 0.2
plt.figure(figsize=(8, 6))
if len(p_b_filtered) > 0:
    plt.hist(p_b_filtered[train_mask], bins=10, range=(0, 1), color='orange', alpha=0.5, label='Training', weights=np.ones_like(p_b_filtered[train_mask]) * 100 / len(train_mask))
    plt.hist(p_b_filtered[val_mask], bins=10, range=(0, 1), color='green', alpha=0.5, label='Validation', weights=np.ones_like(p_b_filtered[val_mask]) * 100 / len(val_mask))
    plt.hist(p_b_filtered[test_mask], bins=10, range=(0, 1), color='blue', alpha=0.5, label='Test', weights=np.ones_like(p_b_filtered[test_mask]) * 100 / len(test_mask))
    plt.title(f'Histogram of p_B for -0.2 < q < 0.2 in {env}')
    plt.xlabel('p_B')
    plt.ylabel('Count (%)')
    plt.legend()
plt.savefig(f'../results/{env}/plots/committor_hist_{env}.png')
plt.close()

# Optional: LIME and SHAP interpretation (for reference, not part of S2/S3)
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_scaled,
    feature_names=[f'CV_{i+1}' for i in range(input_dim)],
    mode='regression'
)
exp_lime = explainer_lime.explain_instance(X_scaled[0], lambda x: predict_p_B(model, x), num_features=input_dim)

explainer_shap = shap.DeepExplainer(model, X_scaled[:100])
shap_values = explainer_shap.shap_values(X_scaled[:100])
mean_shap = np.mean(np.abs(shap_values[0]), axis=0)

print(f"Interpretation complete. Plots saved to {plot_path}")