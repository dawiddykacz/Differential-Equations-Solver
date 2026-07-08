from solvers.models.BasicModel import BasicModel
from solvers.models.ModelWithOptimization import ModelWithOptimizationWrapper

from objects.TrainableVariables import TrainableVariables


class ModelParams:
    def __init__(self, loss, trainable_variables: TrainableVariables, optimizer):
        self.loss = loss
        self.trainable_variables = trainable_variables
        self.optimizer = optimizer


def choose_model(params: ModelParams, with_optimization=False):
    if with_optimization:
        return ModelWithOptimizationWrapper(
            loss=params.loss,
            trainable_variables=params.trainable_variables
        )
    else:
        return BasicModel(
            loss=params.loss,
            trainable_variables=params.trainable_variables,
            optimizer=params.optimizer
        )
