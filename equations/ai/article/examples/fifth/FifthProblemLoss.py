import numpy

from equations.ai.article.examples.fifth.FifthProblem import *

w = 10


class FifthProblemLoss(FifthProblem):
    def __init__(self, space: Space, solution: AISolution = None, weight: float = 10):
        if solution is None:
            super().__init__(SolutionFunction(space, LossSimple()))
        else:
            super().__init__(solution)
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

        x0 = tensorflow.zeros((len(x), 1), dtype=tensorflow.float64)
        x1 = tensorflow.zeros((len(x), 1), dtype=tensorflow.float64) + 1
        y0 = tensorflow.zeros((len(y), 1), dtype=tensorflow.float64)
        y1 = tensorflow.zeros((len(y), 1), dtype=tensorflow.float64) + 1
        return abs(function(x0, y) - y ** 3) + abs(
            function(x1, y) - (1 + y ** 3) * numpy.exp(-1)) + abs(
            function(x, y0) - x * numpy.exp(-x)) + abs(function(x, y1) - numpy.exp(-x) * (x + 1))

    def _condition_weight(self):
        return w
