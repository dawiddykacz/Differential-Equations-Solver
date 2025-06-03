from objects.TaskData import *
from equations.ai.article.examples.third.ThirdProblemSimple import ThirdProblemSimple


class ThirdProblemSimpleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 2)),"Task 5 <0;2>")

    def get_equation(self):
        return ThirdProblemSimple(self.get_space_range().split())
