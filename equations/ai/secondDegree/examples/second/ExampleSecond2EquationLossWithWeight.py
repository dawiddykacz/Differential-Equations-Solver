from equations.ai.secondDegree.examples.second.AbstractExampleSecond2Equation import *


class ExampleSecond2EquationLossWithWeight(AbstractExampleSecond2Equation):
    def __init__(self, space: Space, with_noise: bool, alpha: float, alpha_lower: float):
        trainable_variables = TrainableVariables([1])
        non_trainable_variables = TrainableVariables([1])
        super().__init__(
            SolutionFunction(space,
                             loss_function=LossSimple(
                                 trainable_variables=trainable_variables,
                                 non_trainable_variables=non_trainable_variables,
                                 with_noise=with_noise,
                                 alpha=alpha, alpha_lower=alpha_lower),
                             trainable_variables=trainable_variables,
                             non_trainable_variables=non_trainable_variables))


class LossSimple(Loss):
    def __init__(self, trainable_variables: TrainableVariables,
                 non_trainable_variables: TrainableVariables, with_noise: bool,
                 alpha: float, alpha_lower: float):
        super().__init__(trainable_variables, with_noise)

        self.__non_trainable_variables = non_trainable_variables
        self.__alpha = alpha
        self.__first_alpha = alpha
        self.__alpha_lower = alpha_lower

    def _condition_data_weight(self):
        return self.__non_trainable_variables.get_variables()[0]

    def assign_weights(self, data):
        a = tensorflow.constant(data[0], dtype=tensorflow.float64)
        w = self.__non_trainable_variables.get_variables()[0]
        self.__non_trainable_variables.get_variables()[0] = w * (1 - a) + a

    def recalculate_weights(self, grads_dict, loss_error):
        max_grad_pde = grads_dict['grad_pde_max']
        mean_grad_data = grads_dict['grad_data_mean']

        if not tensorflow.equal(mean_grad_data, 0):
            w = max_grad_pde / mean_grad_data

            self.__non_trainable_variables.get_variables()[0] = ((1 - self.__alpha) *
                                                                 self.__non_trainable_variables.get_variables()[0]
                                                                 + self.__alpha * w)

            if loss_error > 1.05:
                self.__alpha = self.__first_alpha
            else:
                self.__alpha *= self.__alpha_lower
