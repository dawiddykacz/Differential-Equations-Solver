import numpy

from equations.ai.article.examples.third.ThirdProblem import *

w = 1


class ThirdProblemLoss(ThirdProblem):
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
        x = tensorflow.zeros((len(x[0]), 1), dtype=tensorflow.float64)

        with tensorflow.GradientTape(persistent=True) as g:
            g.watch(x)
            y = function(x)
            differential = g.gradient(y, x)

        if differential is None:
            differential = tensorflow.zeros_like(x)
        del g

        return numpy.sqrt((function(x) - 0) ** 2 + (differential - 1) ** 2)

    def _condition_weight(self):
        return w
