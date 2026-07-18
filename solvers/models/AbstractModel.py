import copy

import tensorflow

from objects.TrainableVariables import TrainableVariables


class AbstractModel(tensorflow.keras.Model):
    def __init__(self, loss, trainable_variables: TrainableVariables, dense_list, optimizer=None, custom_layers=None):
        super(AbstractModel, self).__init__()
        if custom_layers is None:
            custom_layers = []
        self.dense_list = tensorflow.keras.Sequential(dense_list)
        self.out_dense = tensorflow.keras.layers.Dense(units=1, activation='linear', dtype='float64')
        self._loss = loss
        self._custom_trainable_variables = trainable_variables
        self.optimizer = optimizer
        self.__grads = None
        self._custom_layers = custom_layers

    def call(self, inputs):
        x = self.dense_list(inputs)
        output = self.out_dense(x)
        return output

    def train_step(self, data=None):
        with tensorflow.GradientTape(persistent=True) as tape:
            loss = self._loss()
            current_loss = loss['loss']
            loss_pde = loss['loss_pde']
            conditions = loss['conditions']
            conditions_data = loss['conditions_data']

        variables_to_train = self.trainable_variables + self._custom_trainable_variables.get_variables()
        grads = tape.gradient(current_loss, variables_to_train)
        self.optimizer.apply_gradients(zip(grads, variables_to_train))

        last_layer_weights = self.trainable_variables

        grad_data = tape.gradient(tensorflow.convert_to_tensor(conditions_data, dtype=tensorflow.float64),
                                  last_layer_weights)
        grad_pde = tape.gradient(tensorflow.convert_to_tensor(loss_pde, dtype=tensorflow.float64), last_layer_weights)
        grad_bc = tape.gradient(tensorflow.convert_to_tensor(conditions, dtype=tensorflow.float64), last_layer_weights)

        del tape

        self.__grads = {
            'grad_data': grad_data,
            'grad_pde': grad_pde,
            'grad_bc': grad_bc,
        }
        return {
            'loss': current_loss
        }

    def get_gradients(self):
        return self.__grads

    def __deepcopy__(self, memo):
        cloned_layers = [
            layer.__class__.from_config(layer.get_config())
            for layer in self.dense_list.layers
        ]
        cloned_custom_layers = [
            layer.__class__.from_config(layer.get_config())
            for layer in self._custom_layers
        ]
        new_model = AbstractModel(loss=copy.deepcopy(self._loss, memo),
                                  trainable_variables=copy.deepcopy(self._custom_trainable_variables, memo),
                                  optimizer=copy.deepcopy(self.optimizer, memo),
                                  dense_list=cloned_layers,
                                  custom_layers=cloned_custom_layers)

        memo[id(self)] = new_model

        return new_model
