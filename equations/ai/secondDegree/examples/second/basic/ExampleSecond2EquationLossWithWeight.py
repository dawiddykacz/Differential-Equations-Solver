import math

from equations.ai.secondDegree.examples.WangHelper import calculate_wang
from equations.ai.secondDegree.examples.second.AbstractExampleSecond2Equation import *


def choose_loss(is_calculating_wang_grads: bool):
    if is_calculating_wang_grads:
        return LossSimpleWithWeightGrads
    return LossSimple


class ExampleSecond2EquationLossWithWeight(AbstractExampleSecond2Equation):
    def __init__(self, space: Space, alpha: float, weight_pde: float = 1,
                 weight_conditions: float = 1, epsilon: float = 1e-8, is_calculating_wang_grads: bool = False):
        non_trainable_variables = TrainableVariables([1], trainable=False)
        super().__init__(
            SolutionFunction(space,
                             loss_function=choose_loss(is_calculating_wang_grads)(
                                 non_trainable_variables=non_trainable_variables,
                                 alpha=alpha,
                                 weight_pde=weight_pde,
                                 weight_conditions=weight_conditions,
                                 eps=epsilon),
                             non_trainable_variables=non_trainable_variables))


class LossSimple(Loss):
    def __init__(self, non_trainable_variables: TrainableVariables,
                 alpha: float, weight_pde: float = 1, weight_conditions: float = 1,
                 eps: float = 1e-8):
        self.__non_trainable_variables = non_trainable_variables
        self.__alpha = alpha
        self.__first_alpha = alpha
        self.weight_pde = weight_pde
        self.weight_conditions = weight_conditions
        self.eps = eps

    def _left_side_of_the_equation(self, function, *x):
        return super()._left_side_of_the_equation(function, *x) / 2

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
        return self.weight_conditions * self.__non_trainable_variables.get_variables()[0]

    def assign_weights(self, data):
        self.__non_trainable_variables.get_variables()[0].assign(tensorflow.constant(data[0], dtype=tensorflow.float64))

    def _pde_weight(self):
        return self.weight_pde

    def recalculate_weights(self, grads_dict, loss_error):
        max_grad_pde = grads_dict['grad_pde_max']
        mean_grad_bc = grads_dict['grad_bc_mean']

        self.__non_trainable_variables.get_variables()[0].assign(calculate_wang(
            mean=mean_grad_bc,
            max=max_grad_pde,
            alpha=self.__alpha,
            eps=self.eps,
            old_value=self.__non_trainable_variables.get_variables()[0]
        ))


class LossSimpleWithWeightGrads(LossSimple):
    def __init__(self, non_trainable_variables: TrainableVariables,
                 alpha: float, weight_pde: float = 1,
                 weight_conditions: float = 1, eps: float = 1e-8):
        super().__init__(
            non_trainable_variables=non_trainable_variables,
            alpha=alpha,
            weight_pde=weight_pde,
            weight_conditions=weight_conditions,
            eps=eps
        )

    def _pde_grad_weight(self):
        return self.weight_pde

    def _condition_grad_weight(self):
        return self.weight_conditions
