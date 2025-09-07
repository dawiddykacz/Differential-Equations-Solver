import tensorflow

from equations.ai.article.examples.fifth.FifthProblemWithDistanceFunction import *

w = 10
class FifthProblemWithPoint(FifthProblem):
    def __init__(self, space: Space, weight: float = 10):
        super().__init__(SolutionFunction(space, LossSimple()))

        global  w
        w = weight


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

        target = tf.exp(tf.cast(-0.5, tf.float64)) * (tf.cast(0.5, tf.float64) + (tf.cast(0.5, tf.float64) ** 3))
        one = tf.ones_like(x)

        dist = (x - one / 2) ** 2 + (y - one / 2) ** 2

        eps = 0.1
        weight = tf.exp(-(dist / eps) ** 2)

        A_xy = (1 - weight) * A_xy + weight * target

        return A_xy + x * (1 - x) * y * (1 - y) * self._ai_solver.calculate(x,y)

class LossSimple(Loss):
    def _condition(self, function, *x):
        y = x[1]
        x = x[0]

        zero = tensorflow.zeros_like(x, dtype=tensorflow.float64)
        one = tensorflow.ones_like(x, dtype=tensorflow.float64)

        return abs(function(zero, y) - y ** 3) + abs(
            function(one, y) - (one + y ** 3) * tf.math.exp(tf.constant(-1,dtype = tf.float64))) + abs(
            function(x, zero) - x * numpy.exp(-x)) + abs(function(x, one) - tf.math.exp(-x) * (x + 1))

    def _condition_weight(self):
        return w
