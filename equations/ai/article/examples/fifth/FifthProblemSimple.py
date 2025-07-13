import numpy
import tensorflow as tf

from equations.ai.article.examples.fifth.FifthProblem import *


class FifthProblemSimple(FifthProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()))

f0 = lambda y: y ** 3
f1 = lambda y: (1 + y ** 3) * tf.exp(tf.cast(-1., tf.float64))
g0 = lambda x: x * tf.exp(-x)
g1 = lambda x: tf.exp(-x) * (x + 1)


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        a = g0(x) - ((1 - x) * g0(tf.cast(0, tf.float64)) + x * g0(tf.cast(1, tf.float64)))
        b = g1(x) - ((1 - x) * g1(tf.cast(0, tf.float64)) + x * g1(tf.cast(1, tf.float64)))
        A_xy = (1 - x) * f0(y) + x * f1(y) + (1 - y) * a + y * b
        return A_xy + x * (1 - x) * y * (1 - y) * self._ai_solver.calculate(x,y)