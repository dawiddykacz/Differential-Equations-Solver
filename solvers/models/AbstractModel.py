import copy

import tensorflow

from objects.TrainableVariables import TrainableVariables


class AbstractModel(tensorflow.keras.Model):
    def __init__(self, loss, trainable_variables: TrainableVariables, dense_list, optimizer = None):
        super(AbstractModel, self).__init__()
        self.dense_list = dense_list
        self.out_dense = tensorflow.keras.layers.Dense(units=1, activation='linear', dtype='float64')
        self._loss = loss
        self._custom_trainable_variables = trainable_variables
        self.optimizer = optimizer

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
        self.optimizer.apply_gradients(zip(grads, variables_to_train))

        return {"loss": current_loss}

    def __deepcopy__(self, memo):
        new_model = AbstractModel(loss=copy.deepcopy(self._loss, memo),
                                  trainable_variables=copy.deepcopy(self._custom_trainable_variables, memo),
                                  optimizer=copy.deepcopy(self.optimizer, memo),
                                  dense_list=self.dense_list)

        memo[id(self)] = new_model

        return new_model
