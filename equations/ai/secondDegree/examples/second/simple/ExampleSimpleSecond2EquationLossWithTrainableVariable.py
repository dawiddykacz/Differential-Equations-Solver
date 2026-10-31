from equations.ai.secondDegree.examples.second.AbstractExampleSecond2EquationWithTrainableVariable import *


class ExampleSimpleSecond2EquationLossWithTrainableVariable(AbstractExampleSecond2EquationWithTrainableVariable):
    def __init__(self, space: Space, with_noise: bool, weight_pde: float = 1,
                 weight_data: float = 1):
        t = TrainableVariables([1])

        super().__init__(
            SimpleSolutionFunction(space, loss_function=LossSimple(t, with_noise,
                                                                   weight_pde=weight_pde,
                                                                   weight_data=weight_data),
                                   trainable_variables=t,
                                   exact_trainable_variables=[0.5]))


class SimpleSolutionFunction(SolutionFunction):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        n = super().calculate(x, y)
        ansatz = -tensorflow.sin(pi * x)
        return (tensorflow.square(x) - one) * (tensorflow.square(y) - one) * n + ansatz


class LossSimple(LossWithTrainableVariable):
    def __init__(self, t: TrainableVariables, with_noise: bool, weight_pde: float = 1,
                 weight_data: float = 1):
        super().__init__(t, with_noise)

        self.weight_pde = weight_pde
        self.weight_data = weight_data

    def _pde_weight(self):
        return self.weight_pde

    def _condition_data_weight(self):
        return self.weight_data
