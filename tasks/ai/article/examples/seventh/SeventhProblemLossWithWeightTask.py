from objects.TaskData import *
from equations.ai.article.examples.seventh.SeventhProblemLossWithWeight import SeventhProblemLossWithWeight


class SeventhProblemLossWithWeightTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1), Range(0, 1)),"7 problem loss (weight)")

    def get_equation(self):
        return SeventhProblemLossWithWeight(self.get_space_range().split())
