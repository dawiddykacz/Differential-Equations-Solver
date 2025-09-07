import tensorflow

from equations.ai.article.examples.fifth.FifthProblemWithDistanceFunction import *

w = 10
class FifthProblemWithPoint(FifthProblem):
    def __init__(self, space: Space, weight: float = 10):
        super().__init__(SolutionFunction(space, LossSimple()))

        global  w
        w = weight


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
