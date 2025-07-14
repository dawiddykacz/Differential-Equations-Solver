from objects.TaskData import *
from equations.ai.article.examples.second.SecondProblemLoss import SecondProblemLoss


class SecondProblemLossTask(TaskData):
    def __init__(self,weight: float = 10):
        super().__init__(SpaceRanges(10, Range(0, 1)),"2 problem loss",weight)

    def get_equation(self):
        return SecondProblemLoss(self.get_space_range().split(),weight=self.get_weight())
