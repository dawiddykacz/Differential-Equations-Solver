from solvers.AISolver import AISolver
from objects.functions.Function import *
from objects.functions.loss.LossFunction import LossFunction
from objects.space.Space import Space
from objects.TrainableVariables import TrainableVariables


class AISolution(Function):
    def __init__(self, space: Space, loss_function: LossFunction, trainable_variables: TrainableVariables
    = TrainableVariables(), non_trainable_variables: TrainableVariables = TrainableVariables(),
                 exact_trainable_variables=None):
        self._ai_solver = AISolver(space, self.calculate, loss_function, trainable_variables,
                                   non_trainable_variables=non_trainable_variables,
                                   calculate_as_numpy=self.calculate_as_numpy)
        self._exact_trainable_variables = exact_trainable_variables

    def calculate(self, *vars):
        return self._ai_solver.calculate(*vars)

    def solve(self, epochs: int, test_points):
        return self._ai_solver.solve(epochs, test_points)

    def get_loss_array(self):
        return self._ai_solver.get_loss_array()

    def get_trainable_variables_array(self):
        return self._ai_solver.get_trainable_variables_array()

    def get_non_trainable_variables_array(self):
        return self._ai_solver.get_non_trainable_variables_array()

    def get_y_by_epoch(self):
        return self._ai_solver.get_y_by_epoch()

    def get_exact_trainable_variables_array(self):
        return self._exact_trainable_variables
