from objects.TaskData import *
from equations.ai.article.examples.third.ThirdProblemLoss import ThirdProblemLoss


class ThirdProblemLossTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 2)),"Task 6 <0;2>")

    def get_equation(self):
        return ThirdProblemLoss(self.get_space_range().split())
