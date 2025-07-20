from objects.TaskData import *
from equations.ai.article.examples.seventh.SeventhProblemSimple import SeventhProblemSimple


class SeventhProblemSimpleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1),Range(0, 1)),"7 problem simple")

    def get_equation(self):
        return SeventhProblemSimple(self.get_space_range().split())