from objects.TaskData import *
from equations.ai.article.examples.fourth.FourthProblemSimple import FourthProblemSimple


class FourthProblemSimpleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(1, Range(0, 1),Range(0, 1)),"Task 9 <0;1>")

    def get_equation(self):
        return FourthProblemSimple(self.get_space_range().split())