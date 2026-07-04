import copy

import tensorflow

from objects.TrainableVariables import TrainableVariables


class BasicModel(tensorflow.keras.Model):
    def __init__(self, loss, trainable_variables: TrainableVariables, optimizer):
        super(BasicModel, self).__init__()
        self.dense_list = [
            tensorflow.keras.layers.Dense(units=10, activation='sigmoid', dtype='float64')
        ]
        self.out_dense = tensorflow.keras.layers.Dense(units=1, activation='linear', dtype='float64')
        self._loss = loss
        self._custom_trainable_variables = trainable_variables
        self._optimizer = optimizer

    def call(self, inputs):
        x = inputs
        for dense in self.dense_list:
            x = dense(x)
        output = self.out_dense(x)
        return output

    def train_step(self, data=None):
        with tensorflow.GradientTape() as tape:
            current_loss = self._loss()

        variables_to_train = self.trainable_variables + self._custom_trainable_variables.get_variables()
        grads = tape.gradient(current_loss, variables_to_train)
        self._optimizer.apply_gradients(zip(grads, variables_to_train))

        return current_loss

    def __deepcopy__(self, memo):
        new_model = BasicModel(loss=copy.deepcopy(self._loss, memo),
                               trainable_variables=copy.deepcopy(self._custom_trainable_variables, memo),
                               optimizer=copy.deepcopy(self._optimizer, memo))

        memo[id(self)] = new_model

        return new_model
