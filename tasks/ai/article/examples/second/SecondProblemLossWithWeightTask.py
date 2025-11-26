from objects.TaskData import *
from equations.ai.article.examples.second.SecondProblemLossWithWeight import SecondProblemLossWithWeight


class SecondProblemLossWithWeightTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 2)),"2 problem loss (weight)")

    def get_equation(self):
        return SecondProblemLossWithWeight(self.get_space_range().split())
