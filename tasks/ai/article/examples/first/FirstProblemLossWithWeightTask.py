from objects.TaskData import *
from equations.ai.article.examples.first.FirstProblemLossWithWeight import FirstProblemLossWithWeight


class FirstProblemLossWithWeightTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"Task 10 <0;1>")

    def get_equation(self):
        return FirstProblemLossWithWeight(self.get_space_range().split())
