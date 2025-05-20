import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adamax

def build_model(n_layers, n_nodes, l2_lambda, input_dim=90):
    model = Sequential()
    for i in range(n_layers):
        model.add(Dense(n_nodes[i], activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda[i]),
                        input_shape=(input_dim,) if i == 0 else None))
    # Output q with tanh activation
    model.add(Dense(1, activation='tanh'))
    # Apply (1 + tanh(q))/2 transformation
    model.add(Lambda(lambda x: (1 + x) / 2))
    model.compile(optimizer=Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                  loss='binary_crossentropy')
    return model

def predict_p_B(model, X):
    p_B = model.predict(X)
    return p_B