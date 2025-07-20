from objects.TaskData import *
from equations.ai.article.examples.seventh.SeventhProblemLoss import SeventhProblemLoss


class SeventhProblemLossTask(TaskData):
    def __init__(self, weight: float = 10):
        super().__init__(SpaceRanges(10, Range(0, 1), Range(0, 1)), "7 problem loss", weight=weight)

    def get_equation(self):
        return SeventhProblemLoss(self.get_space_range().split(), weight=self.get_weight())
