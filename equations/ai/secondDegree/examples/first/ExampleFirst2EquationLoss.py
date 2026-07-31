from equations.ai.secondDegree.examples.first.AbstractExampleFirst2Equation import *


class ExampleFirst2ProblemLoss(AbstractExampleFirst2Problem):
    def __init__(self, space: Space, with_noise: bool, weight: float = 1):
        t = TrainableVariables([1])
        super().__init__(
            SolutionFunction(space, loss_function=LossSimple(t, with_noise, weight), trainable_variables=t))


class LossSimple(Loss):
    def __init__(self, t: TrainableVariables, with_noise: bool, weight: float):
        super().__init__(t, with_noise)
        self._weight = weight

    def _condition_data_weight(self):
        return self._weight
