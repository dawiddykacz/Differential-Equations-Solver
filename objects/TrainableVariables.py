import keras
import tensorflow


class TrainableVariables:
    def __init__(self, variables: [float] = [], trainable: bool = True):
        self._variables = []
        for variable in variables:
            self._variables.append(keras.Variable(variable, dtype='float64', trainable=trainable))

    def get_variables(self):
        return self._variables
