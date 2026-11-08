import tensorflow

from equations.ai.article.examples.first.FirstProblemLoss import *
from equations.ai.secondDegree.examples.WangHelper import calculate_wang


class FirstProblemLossWithWeight(FirstProblemLoss):
    def __init__(self, space: Space, alpha: float = 0.1, alpha_lower: float = 0.9):
        t = TrainableVariables([1])
        super().__init__(None, SolutionFunctionWeight(space, LossSimpleWeight(t, alpha), t))


class SolutionFunctionWeight(SolutionFunction):
    def __init__(self, space: Space, loss_function: LossFunction, t: TrainableVariables):
        super().__init__(space, loss_function, non_trainable_variables=t)


class LossSimpleWeight(LossSimple):
    def __init__(self, t: TrainableVariables, alpha: float):
        self.__t = t
        self.__alpha = alpha
        self.__first_alpha = alpha
        self.eps = 0

    def _condition_weight(self):
        return self.__t.get_variables()[0]

    def recalculate_weights(self, grads_dict, loss_error):
        max_grad_pde = grads_dict['grad_pde_max']
        mean_grad_bc = grads_dict['grad_bc_mean']
        self.__t.get_variables()[0].assign(calculate_wang(
            mean=mean_grad_bc,
            max=max_grad_pde,
            alpha=self.__alpha,
            eps=self.eps,
            old_value=self.__t.get_variables()[0]
        ))
