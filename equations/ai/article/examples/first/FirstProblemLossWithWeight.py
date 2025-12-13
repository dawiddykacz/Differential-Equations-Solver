from equations.ai.article.examples.first.FirstProblemLoss import *

class FirstProblemLossWithWeight(FirstProblemLoss):
    def __init__(self, space: Space):
        t = TrainableVariables([1])
        super().__init__(None,SolutionFunctionWeight(space, LossSimpleWeight(t), t))


class SolutionFunctionWeight(SolutionFunction):
    def __init__(self, space: Space, loss_function: LossFunction, t: TrainableVariables):
        super().__init__(space, loss_function, t)


class LossSimpleWeight(LossSimple):
    def __init__(self, t: TrainableVariables):
        self.__t = t
    def _condition_weight(self):
        return self.__t.get_variables()[0]

    def _add_condition(self):
        return 1 / self.__t.get_variables()[0]
