from objects.TaskData import *
from equations.ai.article.examples.first.FirstProblemLossWithWeight import FirstProblemLossWithWeight


class FirstProblemLossWithWeightTask(TaskData):
    def __init__(self, alpha: float = 0.1, alpha_lower: float = 0.9):
        super().__init__(SpaceRanges(10, Range(0, 1)),f"1 problem loss (weight) "
                                                      f"alpha = {alpha} alpha_lower = {alpha_lower}")
        self.__alpha = alpha
        self.__alpha_lower = alpha_lower

    def get_equation(self):
        return FirstProblemLossWithWeight(self.get_space_range().split(),
                                          alpha=self.__alpha, alpha_lower=self.__alpha_lower)
