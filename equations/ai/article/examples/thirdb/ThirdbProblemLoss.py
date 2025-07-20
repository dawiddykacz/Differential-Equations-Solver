import numpy

from equations.ai.article.examples.thirdb.ThirdbProblem import *

w = 1


class ThirdbProblemLoss(ThirdbProblem):
    def __init__(self, space: Space,solution:AISolution = None,weight:float = 10):
        if solution is None:
            super().__init__(SolutionFunction(space, LossSimple()))
        else:
            super().__init__(solution)
        w = weight


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return self._ai_solver.calculate(x)


class LossSimple(Loss):
    def _condition(self, function, *x):
        x1 = tensorflow.zeros((len(x[0]), 1), dtype=tensorflow.float64)
        x2 = tensorflow.zeros((len(x[0]), 1), dtype=tensorflow.float64) + 1
        return numpy.sqrt((function(x1) - 0) ** 2 + (function(x2) - numpy.sin(1) * numpy.exp(-1 / 5)) ** 2)

    def _condition_weight(self):
        return w
