import tensorflow
import numpy
from tensorflow.math import abs

from equations.ai.secondDegree.third.Third2Equation import *

w = 10


class Third2EquationLoss(Third2Equation):
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
        one_y = tensorflow.ones_like(y, dtype=tensorflow.float64)

        pi = tensorflow.constant(numpy.pi, dtype=tensorflow.float64)

        return abs(function(-1 * one, y) - zero) + abs(function(one, y) - zero) + abs(
            function(x, one_y * -1) + tensorflow.sin(pi * x)) + abs(function(x, one_y) + tensorflow.sin(pi * x))

    def _condition_weight(self):
        return w
