from equations.ai.secondDegree.fourth.Fourth2Equation import *

w = 1


def random_0_01(shape=()):
    return tensorflow.random.uniform(
        shape=shape,
        minval=-0.1,
        maxval=0.1,
        dtype=tensorflow.float64
    )


class Fourth2EquationWithNoiseLoss(Fourth2Equation):
    def __init__(self, space: Space, solution: AISolution = None, weight: float = 1):
        t = TrainableVariables([1])
        if solution is None:
            super().__init__(SolutionFunction(space, LossSimple(t), t))
        else:
            super().__init__(solution)
        global w
        w = weight


class LossSimple(Loss):
    def _condition(self, function, *x):
        loss = super()._condition(function, *x)
        y = x[1]
        x = x[0]

        zero = tensorflow.zeros_like(x, dtype=tensorflow.float64)
        one_x = tensorflow.ones_like(x, dtype=tensorflow.float64)
        one_y = tensorflow.ones_like(y, dtype=tensorflow.float64)

        return loss + abs(function(one_x / 2, one_y / 2) + zero + random_0_01()) + abs(
            function(one_x / 4, one_y / 4) - one_x / 2 + random_0_01()) + abs(
            function(one_x / 4, 3 * one_y / 4) + one_x / 2 + random_0_01()) + abs(
            function(3 * one_x / 4, one_y / 4) - one_x / 2 + random_0_01()) + abs(
            function(3 * one_x / 4, 3 * one_y / 4) + one_x / 2 + random_0_01())


def _condition_weight(self):
    return w
