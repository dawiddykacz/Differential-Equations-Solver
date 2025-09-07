import tensorflow as tf


class ArticleProblemsHelper:
    def __init__(self, f0, f1, g0, g1):
        self.f0 = f0
        self.f1 = f1
        self.g0 = g0
        self.g1 = g1

    def calculate(self, x, y):
        zero = tf.cast(0, tf.float64)
        one = tf.cast(1, tf.float64)
        a = (1 - x) * self.f0(y) + x * self.f1(y) + self.g0(x)
        b = (1 - x) * self.g0(zero) + x * self.g0(one)
        c = self.g1(x) - ((1 - x) * self.g1(zero) + x * self.g1(one))
        return a - b + y * c


class ApproximationFunction:
    def __init__(self, *points, epsilon=0.001):
        self.points = tf.constant(points, dtype=tf.float64)
        self.x_points = self.points[:, :-1]
        self.y_points = self.points[:, -1]
        self.epsilon = epsilon

    def a(self, *variables):
        var = tf.stack(variables, axis=-1)
        var = tf.cast(var, tf.float64)
        diff = self.x_points[None, :, :] - var[:, None, :]
        dists = tf.reduce_sum(diff ** 2, axis=-1)

        soft_min = -self.epsilon * tf.reduce_logsumexp(-dists / self.epsilon, axis=-1)
        return soft_min
