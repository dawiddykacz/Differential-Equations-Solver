from equations.ai.article.examples.second.SecondProblemLoss import *

t = TrainableVariables([2])


class SecondProblemLossWithWeight(SecondProblemLoss):
    def __init__(self, space: Space):
        super().__init__(None,SolutionFunctionWeight(space, LossSimpleWeight()))


class SolutionFunctionWeight(SolutionFunction):
    def __init__(self, space: Space, loss_function: LossFunction):
        super().__init__(space, loss_function, t)


class LossSimpleWeight(LossSimple):
    def _condition_weight(self):
        return t.get_variables()[0]

    def _add_condition(self):
        return 1 / t.get_variables()[0]
