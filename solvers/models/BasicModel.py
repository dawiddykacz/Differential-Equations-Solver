import tensorflow

from solvers.models.AbstractModel import AbstractModel
from objects.TrainableVariables import TrainableVariables
from solvers.models.ModelConfiguration import WangParams
from solvers.models.ModelConfiguration import ModelConfiguration
from solvers.models.WangModel import WangModel


class BasicModel:
    def __init__(self, loss, trainable_variables: TrainableVariables, optimizer, wang_params: WangParams):
        model_configuration = ModelConfiguration()

        if wang_params is None:
            self.__model = AbstractModel(
                loss=loss,
                trainable_variables=trainable_variables,
                optimizer=optimizer,
                dense_list=model_configuration.dense_list
            )
        else:
            self.__model = WangModel(
                loss=loss,
                trainable_variables=trainable_variables,
                optimizer=optimizer,
                layers_Z=model_configuration.dense_list,
                hidden_dim=wang_params.hidden_dim,
                activation=wang_params.activation_function

            )

    def init(self, inputs, epochs: int = 0):
        self.__model(inputs)

    def __call__(self, inputs):
        return self.__model(inputs)

    def get_gradients(self):
        return self.__model.get_gradients()

    def train_step(self, data=None):
        return self.__model.train_step(data=data)
