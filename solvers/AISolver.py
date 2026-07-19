import tensorflow
import numpy

from solvers.models.ChooseModel import ModelParams, choose_model
from objects.space.Space import Space
from objects.functions.loss.LossFunction import LossFunction
from objects.TrainableVariables import TrainableVariables
from solvers.models.ModelConfiguration import ModelConfiguration

_learning_rate = 0.1


def set_learning_rate(learning_rate: float = 0.1):
    global _learning_rate
    if learning_rate <= 0.0:
        raise ValueError("Learning rate must be greater than 0.")
    _learning_rate = learning_rate


class AISolver:
    def __init__(self, space: Space, solution_function, loss_function: LossFunction,
                 trainable_variables: TrainableVariables = TrainableVariables(),
                 non_trainable_variables: TrainableVariables = TrainableVariables(), plots: bool = True):
        self.__points = space.get_points_to_neural_network()
        self.__solution_function = solution_function
        self.__loss_function = loss_function
        self.__plots = plots

        if trainable_variables is None:
            self.__trainable_variables = TrainableVariables()
        else:
            self.__trainable_variables = trainable_variables

        if non_trainable_variables is None:
            self.__non_trainable_variables = TrainableVariables()
        else:
            self.__non_trainable_variables = non_trainable_variables

        model_configuration = ModelConfiguration()
        model_params = ModelParams(loss=self.current_loss,
                                   trainable_variables=trainable_variables,
                                   optimizer=model_configuration.get_optimizer(_learning_rate))

        self.__neural_network = choose_model(params=model_params,
                                             with_optimization=model_configuration.can_optimize(),
                                             wang_params=model_configuration.wang_configuration)

        self.__trainable_plot = []

        for _ in self.__trainable_variables.get_variables():
            self.__trainable_plot.append([])

        self.__non_trainable_plot = []

        for _ in self.__non_trainable_variables.get_variables():
            self.__non_trainable_plot.append([])

        self.__loss_array = numpy.array([])

        if len(self.__points) > 1:
            self.__inputs = tensorflow.concat(self.__points, axis=1)
        else:
            self.__inputs = self.__points[0]

    def calculate(self, *variables):
        inputs = tensorflow.concat(variables, axis=1)
        return self.__neural_network(inputs)

    def current_loss(self):
        return self.__loss_function.calculate(self.__solution_function, *self.__points)

    def solve(self, epochs: int):
        self.__neural_network.init(self.__inputs)

        for i in range(epochs):
            before_loss = self.current_loss()["loss"]
            loss = self.__neural_network.train_step()
            current_loss = loss["loss"]
            loss_error = tensorflow.abs((current_loss - before_loss) / before_loss)
            if self.__plots:
                self.__loss_array = numpy.append(self.__loss_array, current_loss.numpy())

                for j in range(len(self.__trainable_plot)):
                    self.__trainable_plot[j].append(self.__trainable_variables.get_variables()[j].numpy())
                for j in range(len(self.__non_trainable_plot)):
                    self.__non_trainable_plot[j].append(self.__non_trainable_variables.get_variables()[j].numpy())
            self.__loss_function.recalculate_weights(loss, loss_error)

    def get_loss_array(self):
        if self.__plots:
            return self.__loss_array
        return None

    def get_trainable_variables_array(self):
        if self.__plots:
            return self.__trainable_plot
        return None

    def get_non_trainable_variables_array(self):
        if self.__plots:
            return self.__non_trainable_plot
        return None
