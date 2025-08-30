import numpy

from equations.ai.article.examples.seventh.SeventhProblem import *

w = 10


class SeventhProblemLoss(SeventhProblem):
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

        zero = tensorflow.zeros_like(x, dtype=tensorflow.float64)
        one = tensorflow.ones_like(x, dtype=tensorflow.float64)
        one_y = tensorflow.ones_like(x, dtype=tensorflow.float64)

        with tensorflow.GradientTape(persistent=True) as g:
            g.watch(one_y)
            z = function(x, one_y)

            differential_y = g.gradient(z, one_y)

        if differential_y is None:
            differential_y = tensorflow.zeros_like(one_y)
        del g

        return abs(function(zero, y) - zero) + abs(function(one, y) - zero) + abs(
            differential_y - 2 * numpy.sin(numpy.pi * x))

    def _condition_weight(self):
        return w
