import numpy

from equations.ai.article.examples.thirdb.ThirdbProblem import *


class ThirdbProblemLoss(ThirdbProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, LossSimple()))


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return self._ai_solver.calculate(x)

class LossSimple(Loss):
    def _condition(self, function, *x):
        x1 = tensorflow.zeros((len(x[0]), 1), dtype=tensorflow.float64)
        x2 = tensorflow.zeros((len(x[0]), 1), dtype=tensorflow.float64) + 1
        return (function(x1) - 0) + (function(x2) - numpy.sin(1)*numpy.exp(-1/5))