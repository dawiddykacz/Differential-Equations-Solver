import tensorflow

from equations.ai.article.examples.fifth.FifthProblem import *

w = 10


class FifthProblemLoss(FifthProblem):
    def __init__(self, space: Space, solution: AISolution = None, weight: float = 10):
        if solution is None:
            super().__init__(SolutionFunction(space, LossSimple()))
        else:
            super().__init__(solution)
        global w
        w = weight


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]
        return self._ai_solver.calculate(x, y)


class LossSimple(Loss):
    def _condition(self, function, *x):
        y = x[1]
        x = x[0]

        zero = tensorflow.zeros_like(x, dtype=tensorflow.float64)
        one = tensorflow.ones_like(x, dtype=tensorflow.float64)

        return abs(function(zero, y) - y ** 3) + abs(
            function(one, y) - (one + y ** 3) * tensorflow.math.exp(tensorflow.constant(-1,dtype = tensorflow.float64))) + abs(
            function(x, zero) - x * numpy.exp(-x)) + abs(function(x, one) - tensorflow.math.exp(-x) * (x + 1))

    def _condition_weight(self):
        return w
