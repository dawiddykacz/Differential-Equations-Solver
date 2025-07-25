from objects.TaskData import *
from equations.ai.article.examples.fifth.FifthProblemLoss import FifthProblemLoss


class FifthProblemLossTask(TaskData):
    def __init__(self, weight: float = 10):
        super().__init__(SpaceRanges(10, Range(0, 1), Range(0, 1)), "5 problem loss", weight=weight)

    def get_equation(self):
        return FifthProblemLoss(self.get_space_range().split(), weight=self.get_weight())
