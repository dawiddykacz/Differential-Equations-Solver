from equations.ai.secondDegree.examples.second.AbstractExampleSecond2Equation import *
import math

from solvers.WangWeightCalculator import WangWeightCalculator


class ExampleSecond2EquationLossWithWeight(AbstractExampleSecond2Equation):
    def __init__(self, space: Space, with_noise: bool, alpha: float,
                 weight_pde: float = 1,
                 weight_conditions: float = 1,
                 weight_data: float = 1):
        trainable_variables = TrainableVariables([1])
        non_trainable_variables = TrainableVariables([1, 1])
        super().__init__(
            SolutionFunction(space,
                             loss_function=LossSimple(
                                 trainable_variables=trainable_variables,
                                 non_trainable_variables=non_trainable_variables,
                                 with_noise=with_noise,
                                 alpha=alpha,
                                 weight_pde=weight_pde,
                                 weight_conditions=weight_conditions,
                                 weight_data=weight_data),
                             trainable_variables=trainable_variables,
                             non_trainable_variables=non_trainable_variables,
                             exact_trainable_variables=[0.5]))


class LossSimple(Loss):
    def __init__(self, trainable_variables: TrainableVariables,
                 non_trainable_variables: TrainableVariables, with_noise: bool,
                 alpha: float, weight_pde: float = 1,
                 weight_conditions: float = 1, weight_data: float = 1):
        super().__init__(trainable_variables, with_noise)

        self.__non_trainable_variables = non_trainable_variables
        self.__alpha = alpha
        self.__first_alpha = alpha
        self.weight_pde = weight_pde
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

    def _pde_weight(self):
        return self.weight_pde

    def _condition_data_weight(self):
        return self.weight_data * self.__non_trainable_variables.get_variables()[1]

    def _condition_weight(self):
        return self.weight_conditions * self.__non_trainable_variables.get_variables()[0]

    def assign_weights(self, data):
        self.__non_trainable_variables.get_variables()[0].assign(tensorflow.constant(data[0], dtype=tensorflow.float64))
        self.__non_trainable_variables.get_variables()[1].assign(tensorflow.constant(data[1], dtype=tensorflow.float64))

    def recalculate_weights(self, grads_dict, loss_error):
        max_grad_pde = grads_dict['grad_pde_max']
        mean_grad_data = grads_dict['grad_data_mean']
        mean_grad_bc = grads_dict['grad_bc_mean']

        self.__non_trainable_variables.get_variables()[1].assign(WangWeightCalculator.calculate_new_weight(
            alpha=self.__alpha,
            old_weight=self.__non_trainable_variables.get_variables()[1],
            max_grad=max_grad_pde,
            mean_grad=mean_grad_data
        ))
        self.__non_trainable_variables.get_variables()[0].assign(WangWeightCalculator.calculate_new_weight(
            alpha=self.__alpha,
            old_weight=self.__non_trainable_variables.get_variables()[0],
            max_grad=max_grad_pde,
            mean_grad=mean_grad_bc
        ))
