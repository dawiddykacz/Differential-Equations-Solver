from objects.TaskData import *
from equations.ai.article.examples.thirdb.ThirdbProblemLoss import ThirdbProblemLoss


class ThirdbProblemLossTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"3b problem loss")

    def get_equation(self):
        return ThirdbProblemLoss(self.get_space_range().split())