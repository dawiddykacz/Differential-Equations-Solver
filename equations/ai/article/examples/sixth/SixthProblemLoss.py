import numpy

from equations.ai.article.examples.sixth.SixthProblem import *

w = 10


class SixthProblemLoss(SixthProblem):
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
        y1 = tensorflow.zeros((len(y), 1), dtype=tensorflow.float64) + 1

        with tensorflow.GradientTape(persistent=True) as g:
            for point in x:
                g.watch(point)
            z = function(x, y1)
            differential_x, differential_y = g.gradient(z, [x, y])

        if differential_x is None:
            differential_x = tensorflow.zeros_like(x)
        if differential_y is None:
            differential_y = tensorflow.zeros_like(y)
        del g

        return abs(function(x0, y) - 0) + abs(function(x1, y) - 0) + abs(differential_y - 2 * numpy.sin(numpy.pi * x))

    def _condition_weight(self):
        return w
