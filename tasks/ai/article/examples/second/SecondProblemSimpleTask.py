from objects.TaskData import *
from equations.ai.article.examples.second.SecondProblemSimple import SecondProblemSimple


class SecondProblemSimpleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"2 problem simple")

    def get_equation(self):
        return SecondProblemSimple(self.get_space_range().split())
