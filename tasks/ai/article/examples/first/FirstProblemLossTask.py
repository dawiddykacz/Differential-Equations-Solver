from objects.TaskData import *
from equations.ai.article.examples.first.FirstProblemLoss import FirstProblemLoss


class FirstProblemLossTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"Task 1 <0;1>")

    def get_equation(self):
        return FirstProblemLoss(self.get_space_range().split())
