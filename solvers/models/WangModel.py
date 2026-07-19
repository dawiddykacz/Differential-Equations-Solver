import copy
import tensorflow
from objects.TrainableVariables import TrainableVariables
from objects.functions.loss.LossFunction import LossFunction


class WangModel(tensorflow.keras.Model):
    def __init__(self, loss, trainable_variables: TrainableVariables, layers_Z, hidden_dim=10,
                 activation: str = "sigmoid", optimizer=None):
        super(WangModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.activation = activation

        self.encoder_U = tensorflow.keras.layers.Dense(units=hidden_dim, activation=activation, dtype='float64')
        self.encoder_V = tensorflow.keras.layers.Dense(units=hidden_dim, activation=activation, dtype='float64')

        self.layer_H1 = tensorflow.keras.layers.Dense(units=hidden_dim, activation=activation, dtype='float64')

        self.layers_Z = layers_Z

        self.out_dense = tensorflow.keras.layers.Dense(units=1, activation='linear', dtype='float64')

        self._loss = loss
        self._custom_trainable_variables = trainable_variables
        self.optimizer = optimizer

    def call(self, inputs):
        U = self.encoder_U(inputs)
        V = self.encoder_V(inputs)

        H = self.layer_H1(inputs)

        for layer_z in self.layers_Z:
            Z = layer_z(H)
            H = (1.0 - Z) * U + Z * V

        output = self.out_dense(H)
        return output

    @tensorflow.function
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

        return {
            'loss': current_loss,
            'grad_data_mean': LossFunction.mean_abs_grads(grad_data),
            'grad_pde_max': LossFunction.max_abs_grads(grad_pde),
            'grad_bc_mean': LossFunction.mean_abs_grads(grad_bc),
        }

    def __deepcopy__(self, memo):
        new_model = WangModel(
            loss=copy.deepcopy(self._loss, memo),
            trainable_variables=copy.deepcopy(self._custom_trainable_variables, memo),
            hidden_dim=self.hidden_dim,
            layers_Z=copy.deepcopy(self.layers_Z, memo),
            activation=self.activation,
            optimizer=copy.deepcopy(self.optimizer, memo)
        )

        memo[id(self)] = new_model
        return new_model
