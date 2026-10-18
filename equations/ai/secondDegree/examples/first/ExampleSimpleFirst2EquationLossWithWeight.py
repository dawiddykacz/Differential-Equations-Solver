from equations.ai.secondDegree.examples.first.AbstractExampleFirst2Equation import *
from solvers.WangWeightCalculator import WangWeightCalculator


class ExampleSimpleFirst2EquationLossWithWeight(AbstractExampleFirst2Problem):
    def __init__(self, space: Space, with_noise: bool, alpha: float, weight_pde: float = 1, weight_data: float = 1):
        trainable_variables = TrainableVariables([1])
        non_trainable_variables = TrainableVariables([1])
        super().__init__(
            SimpleSolutionFunction(space,
                                   loss_function=LossSimple(
                                       trainable_variables=trainable_variables,
                                       non_trainable_variables=non_trainable_variables,
                                       with_noise=with_noise,
                                       alpha=alpha,
                                       weight_pde=weight_pde,
                                       weight_data=weight_data),
                                   trainable_variables=trainable_variables,
                                   non_trainable_variables=non_trainable_variables,
                                   exact_trainable_variables=[0.5]))


class SimpleSolutionFunction(SolutionFunction):
    def calculate(self, *vars):
        x = vars[0]
        return self._ai_solver.calculate(x) * x


class LossSimple(Loss):
    def __init__(self, trainable_variables: TrainableVariables,
                 non_trainable_variables: TrainableVariables, with_noise: bool,
                 alpha: float, weight_pde: float = 1, weight_data: float = 1):
        super().__init__(trainable_variables, with_noise)

        self.__non_trainable_variables = non_trainable_variables
        self.__alpha = alpha
        self.__first_alpha = alpha
        self.weight_data = weight_data
        self.weight_pde = weight_pde

    def _condition_data_weight(self):
        return self.weight_data * self.__non_trainable_variables.get_variables()[0]

    def assign_weights(self, data):
        self.__non_trainable_variables.get_variables()[0].assign(tensorflow.constant(data[0], dtype=tensorflow.float64))

    def _pde_weight(self):
        return self.weight_pde

    def recalculate_weights(self, grads_dict, loss_error):
        max_grad_data_pde = grads_dict['grad_pde_max']
        mean_grad_data = grads_dict['grad_data_mean']

        self.__non_trainable_variables.get_variables()[0].assign(WangWeightCalculator.calculate_new_weight(
            alpha=self.__alpha,
            old_weight=self.__non_trainable_variables.get_variables()[0],
            max_grad=max_grad_data_pde,
            mean_grad=mean_grad_data
        ))
