from objects.TaskData import *
from equations.ai.article.examples.fifth.FifthProblemWithAddedPoint import FifthProblemWithPoint


class FifthProblemLossWithPointTask(TaskData):
    def __init__(self, weight: float = 10):
        super().__init__(SpaceRanges(10, Range(0, 1), Range(0, 1)), "5 problem with point loss", weight=weight)

    def get_equation(self):
        return FifthProblemWithPoint(self.get_space_range().split(), weight=self.get_weight())
