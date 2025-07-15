from objects.TaskData import *
from equations.ai.article.examples.sixth.SixthProblemSimple import SixthProblemSimple


class SixthProblemSimpleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1),Range(0, 1)),"6 problem simple")

    def get_equation(self):
        return SixthProblemSimple(self.get_space_range().split())