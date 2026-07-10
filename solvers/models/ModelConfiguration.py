import tensorflow

def get_basic_dense():
    dense_list = [
        tensorflow.keras.layers.Dense(units=10, activation='sigmoid', dtype='float64')
    ]