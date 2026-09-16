from equations.ai.secondDegree.examples.first.AbstractExampleFirst2Equation import *


class ExampleFirst2ProblemLoss(AbstractExampleFirst2Problem):
    def __init__(self, space: Space, with_noise: bool, weight_conditions: float = 1, weight_data: float = 1):
        t = TrainableVariables([1])
        super().__init__(
            SolutionFunction(space, loss_function=LossSimple(t, with_noise, weight_conditions, weight_data),
                             trainable_variables=t, exact_trainable_variables=[0.5]))


class LossSimple(Loss):
    def __init__(self, t: TrainableVariables, with_noise: bool, weight_conditions: float = 1, weight_data: float = 1):
        super().__init__(t, with_noise)

        self.weight_conditions = weight_conditions
        self.weight_data = weight_data

    def _condition(self, function, *x):

        zero = tensorflow.zeros_like(x[0], dtype=tensorflow.float64)

        return function(zero) - zero

    def _condition_weight(self):
        return self.weight_conditions

    def _condition_data_weight(self):
        return self.weight_data
