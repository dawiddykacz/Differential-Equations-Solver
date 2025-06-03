import tensorflow

class TrainableVariables:
    def __init__(self, variables:[float] = []):
        self._variables = []
        for variable in variables:
            self._variables.append(tensorflow.Variable(variable, dtype=tensorflow.float64, trainable=True))

    def get_variables(self):
        return self._variables