import tensorflow

from solvers.models.AbstractModel import AbstractModel
from objects.TrainableVariables import TrainableVariables
from solvers.models.ModelConfiguration import ModelConfiguration


class BasicModel:
    def __init__(self, loss, trainable_variables: TrainableVariables, optimizer):
        model_configuration = ModelConfiguration()

        self.__model = AbstractModel(
            loss=loss,
            trainable_variables=trainable_variables,
            optimizer=optimizer,
            dense_list=model_configuration.dense_list
        )

    def init(self, inputs, epochs: int = 0):
        self.__model(inputs)

    def __call__(self, inputs):
        return self.__model(inputs)

    def train_step(self, data=None):
        return self.__model.train_step(data=data)
