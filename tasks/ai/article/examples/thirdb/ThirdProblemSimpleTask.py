from objects.TaskData import *
from equations.ai.article.examples.thirdb.ThirdbProblemSimple import ThirdbProblemSimple


class ThirdbProblemSimpleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"Task 7 <0;1>")

    def get_equation(self):
        return ThirdbProblemSimple(self.get_space_range().split())