from objects.TaskData import *
from equations.ai.article.examples.sixth.SixthProblemLoss import SixthProblemLoss


class SixthProblemLossTask(TaskData):
    def __init__(self, weight: float = 10):
        super().__init__(SpaceRanges(10, Range(0, 1), Range(0, 1)), "6 problem loss", weight=weight)

    def get_equation(self):
        return SixthProblemLoss(self.get_space_range().split(), weight=self.get_weight())
