from equations.ai.article.examples.fifth.FifthProblemLoss import *

class FifthProblemLossWithWeight(FifthProblemLoss):
    def __init__(self, space: Space):
        t = TrainableVariables([1])
        super().__init__(None, SolutionFunctionWeight(space, LossSimpleWeight(t),t))


class SolutionFunctionWeight(SolutionFunction):
    def __init__(self, space: Space, loss_function: LossFunction,t: TrainableVariables):
        super().__init__(space, loss_function, t)


class LossSimpleWeight(LossSimple):
    def __init__(self,t: TrainableVariables):
        self.t = t

    def _condition_weight(self):
        return self.t.get_variables()[0]

    def _add_condition(self):
        return 1 / self.t.get_variables()[0]
