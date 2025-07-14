from objects.TaskData import *
from equations.ai.article.examples.third.ThirdProblemLossWithWeight import ThirdProblemLossWithWeight


class ThirdProblemLossWithWeightTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 2)),"3 problem loss (weight)")

    def get_equation(self):
        return ThirdProblemLossWithWeight(self.get_space_range().split())
