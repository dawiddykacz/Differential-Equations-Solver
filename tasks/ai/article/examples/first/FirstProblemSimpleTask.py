from objects.TaskData import *
from equations.ai.article.examples.first.FirstProblemSimple import FirstProblemSimple


class FirstProblemSimpleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"Task 2 <0;1>")

    def get_equation(self):
        return FirstProblemSimple(self.get_space_range().split())
