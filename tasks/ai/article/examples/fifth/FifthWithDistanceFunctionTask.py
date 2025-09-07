from objects.TaskData import *
from equations.ai.article.examples.fifth.FifthProblemWithDistanceFunction import FifthProblemWithDistanceFunction


class FifthProblemWithDistanceFunctionTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1),Range(0, 1)),"5 problem simple with distance function")

    def get_equation(self):
        return FifthProblemWithDistanceFunction(self.get_space_range().split())