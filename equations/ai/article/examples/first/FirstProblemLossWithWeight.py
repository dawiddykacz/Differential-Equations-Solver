from equations.ai.article.examples.first.FirstProblemLoss import *

t = TrainableVariables([1])


class FirstProblemLossWithWeight(FirstProblemLoss):
    def __init__(self, space: Space):
        super().__init__(None,SolutionFunctionWeight(space, LossSimpleWeight()))


class SolutionFunctionWeight(SolutionFunction):
    def __init__(self, space: Space, loss_function: LossFunction):
        super().__init__(space, loss_function, t)


class LossSimpleWeight(LossSimple):
    def _condition_weight(self):
        return t.get_variables()[0]
