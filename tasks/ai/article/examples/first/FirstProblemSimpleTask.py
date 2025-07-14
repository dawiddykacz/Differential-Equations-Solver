from objects.TaskData import *
from equations.ai.article.examples.first.FirstProblemSimple import FirstProblemSimple


class FirstProblemSimpleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"1 problem simple")

    def get_equation(self):
        return FirstProblemSimple(self.get_space_range().split())
