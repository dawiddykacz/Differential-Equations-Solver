from objects.TaskData import *
from equations.ai.article.examples.second.SecondProblemLoss import SecondProblemLoss


class SecondProblemLossTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"Task 3 <0;1>")

    def get_equation(self):
        return SecondProblemLoss(self.get_space_range().split())
