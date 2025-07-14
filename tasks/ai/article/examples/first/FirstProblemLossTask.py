from objects.TaskData import *
from equations.ai.article.examples.first.FirstProblemLoss import FirstProblemLoss


class FirstProblemLossTask(TaskData):
    def __init__(self,weight: float = 10):
        super().__init__(SpaceRanges(10, Range(0, 1)),"1 problem loss",weight)

    def get_equation(self):
        return FirstProblemLoss(self.get_space_range().split(),weight=self.get_weight())
