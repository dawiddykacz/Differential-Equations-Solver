from equations.ai.secondDegree.examples.first.AbstractExampleFirst2Equation import *


class ExampleFirst2EquationLossWithWeight(AbstractExampleFirst2Problem):
    def __init__(self, space: Space, with_noise: bool, alpha: float, weight_conditions: float = 1,
                 weight_pde: float = 1, weight_data: float = 1):
        trainable_variables = TrainableVariables([1])
        non_trainable_variables = TrainableVariables([1, 1], trainable=False)
        super().__init__(
            SolutionFunction(space,
                             loss_function=LossSimple(
                                 trainable_variables=trainable_variables,
                                 non_trainable_variables=non_trainable_variables,
                                 with_noise=with_noise,
                                 alpha=alpha, weight_conditions=weight_conditions,
                                 weight_pde=weight_pde,
                                 weight_data=weight_data),
                             trainable_variables=trainable_variables,
                             non_trainable_variables=non_trainable_variables,
                             exact_trainable_variables=[0.5]))


class LossSimple(Loss):
    def __init__(self, trainable_variables: TrainableVariables,
                 non_trainable_variables: TrainableVariables, with_noise: bool,
                 alpha: float, weight_pde: float = 1, weight_conditions: float = 1,
                 weight_data: float = 1):
        super().__init__(trainable_variables, with_noise)

        self.__non_trainable_variables = non_trainable_variables
        self.__alpha = alpha
        self.__first_alpha = alpha
        self.weight_conditions = weight_conditions
        self.weight_data = weight_data
        self.weight_pde = weight_pde

    def _condition(self, function, *x):

        zero = tensorflow.zeros_like(x[0], dtype=tensorflow.float64)

        return function(zero) - zero

    def _condition_data_weight(self):
        tensorflow.print("d", self.weight_data, self.__non_trainable_variables.get_variables()[1])
        return self.weight_data * self.__non_trainable_variables.get_variables()[1]

    def _condition_weight(self):
        tensorflow.print("c", self.weight_conditions, self.__non_trainable_variables.get_variables()[0])
        return self.weight_conditions * self.__non_trainable_variables.get_variables()[0]

    def _pde_weight(self):
        return self.weight_pde

    def assign_weights(self, data):
        self.__non_trainable_variables.get_variables()[0].assign(tensorflow.constant(data[0], dtype=tensorflow.float64))
        self.__non_trainable_variables.get_variables()[1].assign(tensorflow.constant(data[1], dtype=tensorflow.float64))

    def recalculate_weights(self, grads_dict, loss_error):
        max_grad_pde = grads_dict['grad_pde_max']
        mean_grad_data = grads_dict['grad_data_mean']
        mean_grad_bc = grads_dict['grad_bc_mean']
        eps = 1e-8

        w = max_grad_pde / (mean_grad_data + eps)

        self.__non_trainable_variables.get_variables()[1].assign((1 - self.__alpha) *
                                                                 self.__non_trainable_variables.get_variables()[1]
                                                                 + self.__alpha * w)
        w2 = max_grad_pde / (mean_grad_bc + eps)

        self.__non_trainable_variables.get_variables()[0].assign((1 - self.__alpha) *
                                                                 self.__non_trainable_variables.get_variables()[0]
                                                                 + self.__alpha * w2)
