from objects.TaskData import *
from equations.ai.article.examples.sixth.SixthProblemLossWithWeight import SixthProblemLossWithWeight


class SixthProblemLossWithWeightTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1), Range(0, 1)),"6 problem loss (weight)")

    def get_equation(self):
        return SixthProblemLossWithWeight(self.get_space_range().split())
