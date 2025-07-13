from objects.TaskData import *
from equations.ai.article.examples.fifth.FifthProblemSimple import FifthProblemSimple


class FifthProblemSimpleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1),Range(0, 1)),"5 problem simple")

    def get_equation(self):
        return FifthProblemSimple(self.get_space_range().split())