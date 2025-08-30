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
        zero = tensorflow.zeros_like(x[0], dtype=tensorflow.float64)
        one = tensorflow.ones_like(x[0], dtype=tensorflow.float64)

        with tensorflow.GradientTape(persistent=True) as g:
            g.watch(zero)
            y = function(zero)
            differential = g.gradient(y, zero)

        if differential is None:
            differential = tensorflow.zeros_like(x)
        del g

        return tensorflow.abs(function(zero) - zero) + tensorflow.abs(differential - one)

    def _condition_weight(self):
        return w
