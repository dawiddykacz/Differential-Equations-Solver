import tensorflow
import math

from equations.ai.secondDegree.examples.second.AbstractExampleSecond2Equation import *


class ExampleSecond2EquationLoss(AbstractExampleSecond2Equation):
    def __init__(self, space: Space, with_noise: bool, weight_conditions: float = 1, weight_data: float = 1):
        t = TrainableVariables([1])

        super().__init__(
            SolutionFunction(space, loss_function=LossSimple(t, with_noise,
                                                             weight_conditions=weight_conditions,
                                                             weight_data=weight_data), trainable_variables=t,
                             exact_trainable_variables=[0.5]))


class LossSimple(Loss):
    def __init__(self, t: TrainableVariables, with_noise: bool, weight_conditions: float = 1, weight_data: float = 1):
        super().__init__(t, with_noise)

        self.weight_conditions = weight_conditions
        self.weight_data = weight_data

    @tensorflow.function
    def _condition(self, function, *args):
        x, y = args[0], args[1]

        ones_x = tensorflow.ones_like(x, dtype=tensorflow.float64)
        ones_y = tensorflow.ones_like(y, dtype=tensorflow.float64)
        minus_ones_x = -ones_x
        minus_ones_y = -ones_y

        w1 = function(minus_ones_x, y)
        w2 = function(ones_x, y)

        pi = tensorflow.constant(math.pi, dtype=tensorflow.float64)
        target = -tensorflow.sin(pi * x)

        w3 = function(x, minus_ones_y) - target
        w4 = function(x, ones_y) - target

        return tensorflow.abs(w1) + tensorflow.abs(w2) + tensorflow.abs(w3) + tensorflow.abs(w4)

    def _condition_weight(self):
        return self.weight_conditions

    def _condition_data_weight(self):
        return self.weight_data
