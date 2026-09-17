from equations.ai.secondDegree.examples.second.AbstractExampleSecond2Equation import *


class ExampleSimpleSecond2EquationLossWithWeight(AbstractExampleSecond2Equation):
    def __init__(self, space: Space, with_noise: bool, alpha: float, weight_data: float = 1):
        trainable_variables = TrainableVariables([1])
        non_trainable_variables = TrainableVariables([1, 1])
        super().__init__(
            SimpleSolutionFunction(space,
                                   loss_function=LossSimple(
                                       trainable_variables=trainable_variables,
                                       non_trainable_variables=non_trainable_variables,
                                       with_noise=with_noise,
                                       alpha=alpha, weight_data=weight_data),
                                   trainable_variables=trainable_variables,
                                   non_trainable_variables=non_trainable_variables,
                                   exact_trainable_variables=[0.5]))


class SimpleSolutionFunction(SolutionFunction):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        n = super().calculate(x, y)
        ansatz = -tensorflow.sin(pi * x)
        return (tensorflow.square(x) - one) * (tensorflow.square(y) - one) * n + ansatz


class LossSimple(Loss):
    def __init__(self, trainable_variables: TrainableVariables,
                 non_trainable_variables: TrainableVariables, with_noise: bool,
                 alpha: float, weight_data: float = 1):
        super().__init__(trainable_variables, with_noise)

        self.__non_trainable_variables = non_trainable_variables
        self.__alpha = alpha
        self.__first_alpha = alpha
        self.weight_data = weight_data

    def _condition_data_weight(self):
        return self.weight_data * self.__non_trainable_variables.get_variables()[0]

    def assign_weights(self, data):
        self.__non_trainable_variables.get_variables()[0] = tensorflow.constant(data[0], dtype=tensorflow.float64)

    def recalculate_weights(self, grads_dict, loss_error):
        max_grad_pde = grads_dict['grad_pde_max']
        mean_grad_data = grads_dict['grad_data_mean']

        if not tensorflow.equal(mean_grad_data, 0):
            w2 = max_grad_pde / mean_grad_data

            self.__non_trainable_variables.get_variables()[0] = ((1 - self.__alpha) *
                                                                 self.__non_trainable_variables.get_variables()[0]
                                                                 + self.__alpha * w2)
