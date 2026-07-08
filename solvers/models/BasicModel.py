import tensorflow

from solvers.models.AbstractModel import AbstractModel
from objects.TrainableVariables import TrainableVariables


class BasicModel:
    def __init__(self, loss, trainable_variables: TrainableVariables, optimizer):
        dense_list = [
            #tensorflow.keras.layers.Dense(units=11, activation='sigmoid', dtype='float64'),
            #tensorflow.keras.layers.Dense(units=10, activation='sigmoid', dtype='float64'),
            tensorflow.keras.layers.Dense(units=10, activation='sigmoid', dtype='float64')
        ]

        self.__model = AbstractModel(
            loss=loss,
            trainable_variables=trainable_variables,
            optimizer=optimizer,
            dense_list=dense_list
        )

    def init(self, inputs, epochs: int = 0):
        self.__model(inputs)

    def __call__(self, inputs):
        return self.__model(inputs)

    def train_step(self, data=None):
        return self.__model.train_step(data=data)
