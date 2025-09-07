import tensorflow
import tensorflow as tf

from equations.ai.article.examples.fifth.FifthProblem import *
from helpers.EquationsHelper import ApproximationFunction

approximation_function = ApproximationFunction([0, 0, 0], [0, 1, 1], [1, 0, numpy.exp(-1)], [1, 1, 2 * numpy.exp(-1)],
                                               [0.5, 0.5, 1])


class FifthProblemWithDistanceFunction(FifthProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()))


f0 = lambda y: y ** 3
f1 = lambda y: (1 + y ** 3) * tf.exp(tf.cast(-1., tf.float64))
g0 = lambda x: x * tf.exp(-x)
g1 = lambda x: tf.exp(-x) * (x + 1)


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        global approximation_function
        x = vars[0]
        y = vars[1]

        a = g0(x) - ((1 - x) * g0(tf.cast(0, tf.float64)) + x * g0(tf.cast(1, tf.float64)))
        b = g1(x) - ((1 - x) * g1(tf.cast(0, tf.float64)) + x * g1(tf.cast(1, tf.float64)))
        A_xy = (1 - x) * f0(y) + x * f1(y) + (1 - y) * a + y * b

        target = tf.exp(tf.cast(-0.5, tf.float64)) * (tf.cast(0.5, tf.float64) + (tf.cast(0.5, tf.float64) ** 3))
        one = tf.ones_like(x)

        dist = (x - one / 2) ** 2 + (y - one / 2) ** 2

        eps = 0.1
        weight = tf.exp(-(dist / eps) ** 2)

        A_xy = (1 - weight) * A_xy + weight * target

        return approximation_function.a(x, y) * self._ai_solver.calculate(x, y) + A_xy
