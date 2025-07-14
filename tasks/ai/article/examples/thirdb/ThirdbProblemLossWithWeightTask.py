from objects.TaskData import *
from equations.ai.article.examples.thirdb.ThirdbProblemLossWithWeight import ThirdbProblemLossWithWeight


class ThirdbProblemLossWithWeightTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"3b problem loss (weight)")

    def get_equation(self):
        return ThirdbProblemLossWithWeight(self.get_space_range().split())
