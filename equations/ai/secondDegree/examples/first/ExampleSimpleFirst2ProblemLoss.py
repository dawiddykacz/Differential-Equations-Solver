from equations.ai.secondDegree.examples.first.AbstractExampleFirst2Equation import *


class ExampleSimpleFirst2ProblemLoss(AbstractExampleFirst2Problem):
    def __init__(self, space: Space, with_noise: bool, weight_data: float = 1):
        t = TrainableVariables([1])
        super().__init__(
            SimpleSolutionFunction(space, loss_function=LossSimple(t, with_noise, weight_data),
                                   trainable_variables=t, exact_trainable_variables=[0.5]))


class SimpleSolutionFunction(SolutionFunction):
    def calculate(self, *vars):
        x = vars[0]
        return self._ai_solver.calculate(x) * x


class LossSimple(Loss):
    def __init__(self, t: TrainableVariables, with_noise: bool, weight_data: float = 1):
        super().__init__(t, with_noise)

        self.weight_data = weight_data

    def _condition_data_weight(self):
        return self.weight_data
