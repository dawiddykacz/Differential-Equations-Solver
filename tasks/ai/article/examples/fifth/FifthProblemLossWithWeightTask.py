from objects.TaskData import *
from equations.ai.article.examples.fifth.FifthProblemLossWithWeight import FifthProblemLossWithWeight


class FifthProblemLossWithWeightTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1), Range(0, 1)),"5 problem loss (weight)")

    def get_equation(self):
        return FifthProblemLossWithWeight(self.get_space_range().split())
