import tensorflow

from equations.ai.article.examples.first.FirstProblemLoss import *


class FirstProblemLossWithWeight(FirstProblemLoss):
    def __init__(self, space: Space, alpha: float = 0.1, alpha_lower: float = 0.9):
        t = TrainableVariables([1])
        super().__init__(None, SolutionFunctionWeight(space, LossSimpleWeight(t, alpha, alpha_lower), t))


class SolutionFunctionWeight(SolutionFunction):
    def __init__(self, space: Space, loss_function: LossFunction, t: TrainableVariables):
        super().__init__(space, loss_function, non_trainable_variables=t)


class LossSimpleWeight(LossSimple):
    def __init__(self, t: TrainableVariables, alpha: float, alpha_lower: float):
        self.__t = t
        self.__alpha = alpha
        self.__first_alpha = alpha
        self.__alpha_lower = alpha_lower

    def _condition_weight(self):
        return self.__t.get_variables()[0]

    def recalculate_weights(self, grads_dict, loss_error):
        max_grad_pde = grads_dict['grad_pde_max']
        mean_grad_bc = grads_dict['grad_bc_mean']
        if not tensorflow.equal(mean_grad_bc, 0):
            w = max_grad_pde / mean_grad_bc

            self.__t.get_variables()[0] = (1 - self.__alpha) * self.__t.get_variables()[0] + self.__alpha * w

            if loss_error > 1.05:
                self.__alpha = self.__first_alpha
            else:
                self.__alpha *= self.__alpha_lower
