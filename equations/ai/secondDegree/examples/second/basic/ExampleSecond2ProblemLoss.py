import math

from equations.ai.secondDegree.examples.second.AbstractExampleSecond2Equation import *


class ExampleSecond2ProblemLoss(AbstractExampleSecond2Equation):
    def __init__(self, space: Space, weight_pde: float = 1, weight_conditions: float = 1):
        super().__init__(
            SolutionFunction(space, loss_function=LossSimple(weight_pde=weight_pde,
                                                             weight_conditions=weight_conditions)
                             ))


class LossSimple(Loss):
    def __init__(self, weight_pde: float = 1, weight_conditions: float = 1):
        self.weight_conditions = weight_conditions
        self.weight_pde = weight_pde

    def _left_side_of_the_equation(self, function, *x):
        return super()._left_side_of_the_equation(function, *x) / 2

    @tensorflow.function
    def _condition(self, function, *args):
        x, y = args[0], args[1]

        ones_x = tensorflow.ones_like(x, dtype=tensorflow.float64)
        ones_y = tensorflow.ones_like(y, dtype=tensorflow.float64)
        minus_ones_x = -ones_x
        minus_ones_y = -ones_y

        w1 = function(minus_ones_x, y)
        w2 = function(ones_x, y)

        pi = tensorflow.constant(math.pi, dtype=tensorflow.float64)
        target = -tensorflow.sin(pi * x)

        w3 = function(x, minus_ones_y) - target
        w4 = function(x, ones_y) - target

        return tensorflow.abs(w1) + tensorflow.abs(w2) + tensorflow.abs(w3) + tensorflow.abs(w4)

    def _pde_weight(self):
        return self.weight_pde

    def _condition_weight(self):
        return self.weight_conditions
