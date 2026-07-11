import tensorflow

from equations.ai.article.examples.first.FirstProblemLoss import *




class FirstProblemLossWithWeight(FirstProblemLoss):
    def __init__(self, space: Space):
        t = TrainableVariables([1])
        super().__init__(None, SolutionFunctionWeight(space, LossSimpleWeight(t), t))


class SolutionFunctionWeight(SolutionFunction):
    def __init__(self, space: Space, loss_function: LossFunction, t: TrainableVariables):
        super().__init__(space, loss_function, non_trainable_variables=t)


class LossSimpleWeight(LossSimple):
    def __init__(self, t: TrainableVariables):
        self.__t = t
        self.__alpha = 0.1

    def _condition_weight(self):
        return self.__t.get_variables()[0]

    def recalculate_weights(self, grads_dict):
        grad_pde = grads_dict['grad_pde']
        grad_bc = grads_dict['grad_bc']
        if all(g is not None for g in [grad_pde[0], grad_bc[0]]):
            max_grad_pde = LossFunction.max_abs_grads(grad_pde)
            mean_grad_bc = LossFunction.mean_abs_grads(grad_bc)
            w = max_grad_pde / (mean_grad_bc + 1e-8)
            self.__t.get_variables()[0] = (1 - self.__alpha) * self.__t.get_variables()[0] + self.__alpha * w
            self.__alpha *= 0.9
