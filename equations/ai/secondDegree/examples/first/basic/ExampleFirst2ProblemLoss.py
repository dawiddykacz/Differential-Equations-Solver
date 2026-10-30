from equations.ai.secondDegree.examples.first.AbstractExampleFirst2Equation import *


class ExampleFirst2ProblemLoss(AbstractExampleFirst2Problem):
    def __init__(self, space: Space, weight_pde: float = 1, weight_conditions: float = 1):
        super().__init__(
            SolutionFunction(space, loss_function=LossSimple(weight_pde=weight_pde,
                                                             weight_conditions=weight_conditions)
                             ))


class LossSimple(Loss):
    def __init__(self, weight_pde: float = 1, weight_conditions: float = 1):
        self.weight_conditions = weight_conditions
        self.weight_pde = weight_pde

    def _left_side_of_the_equation(self, function, *x):
        return super()._left_side_of_the_equation(function, *x) / 2

    def _condition(self, function, *x):
        zero = tensorflow.zeros_like(x[0], dtype=tensorflow.float64)

        return function(zero) - zero

    def _pde_weight(self):
        return self.weight_pde

    def _condition_weight(self):
        return self.weight_conditions
