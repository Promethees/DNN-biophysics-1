import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adamax

def build_model(n_layers, n_nodes, l2_lambda):
    model = Sequential()
    for i in range(n_layers):
        model.add(Dense(n_nodes[i], activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda[i]),
                        input_shape=(90,) if i == 0 else None))
    model.add(Dense(1, activation='tanh'))  # Output q, later transformed to p_B
    model.compile(optimizer=Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                  loss='binary_crossentropy')
    return model

def predict_p_B(model, X):
    q = model.predict(X)
    p_B = (1 + q) / 2  # Sigmoidal transformation
    return p_B