import math

from equations.ai.secondDegree.examples.navier.AbstractNavierEquation import *


class ExampleSecond2ProblemLoss(AbstractNavierEquation):
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
        return super()._left_side_of_the_equation(function, *x)

    @tensorflow.function
    def _condition(self, function, *args):
        x, y = args[0], args[1]

        # Warunki brzegowe w dziedzinie [-1, 1] x [-1, 1]
        ones_x = tensorflow.ones_like(x, dtype=tensorflow.float64)
        ones_y = tensorflow.ones_like(y, dtype=tensorflow.float64)
        minus_ones_x = -ones_x
        minus_ones_y = -ones_y

        w1 = function(minus_ones_x, y)
        w2 = function(ones_x, y)
        w3 = function(x, minus_ones_y)
        w4 = function(x, ones_y)

        # W równaniu Helmholtza z rozwiązania dokładnego sin(pi*x)*sin(4*pi*y)
        # wynika, że funkcja zeruje się na brzegach dziedziny (-1, 1)[cite: 4]
        target = tensorflow.zeros_like(x, dtype=tensorflow.float64)

        return tensorflow.concat([w1 - target, w2 - target, w3 - target, w4 - target], axis=0)

    def _pde_weight(self):
        return self.weight_pde

    def _condition_weight(self):
        return self.weight_conditions
