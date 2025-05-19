import lime.lime_tabular
import shap
import pandas as pd
import numpy as np
from model import predict_p_B
from utils import load_data, preprocess_data
from tensorflow.keras.models import load_model

# Load data and model
X, y = load_data('../data/vacuum_data.csv')  # Or water_data.csv
X_scaled, scaler = preprocess_data(X)
model = load_model('../results/models/final_model.h5')

# LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_scaled,
    feature_names=[f'CV_{i}' for i in range(90)],
    mode='regression'
)
# Example: Explain a single instance
instance = X_scaled[0]
exp = explainer.explain_instance(instance, lambda x: predict_p_B(model, x), num_features=10)
exp.save_to_file('../results/plots/lime_explanation.html')

# SHAP
explainer = shap.DeepExplainer(model, X_scaled[:100])  # Use a subset for background
shap_values = explainer.shap_values(X_scaled[:100])
shap.summary_plot(shap_values, X_scaled[:100], feature_names=[f'CV_{i}' for i in range(90)],
                  show=False)
import matplotlib.pyplot as plt
plt.savefig('../results/plots/shap_summary.png')