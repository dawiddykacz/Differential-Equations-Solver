from objects.TaskData import *
from equations.ai.secondDegree.second.Second2EquationLossWithNoise import Second2ProblemLossWithNoise


class Second2ProblemLossWithNoiseTask(TaskData):
    def __init__(self,weight:float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1)),"2 second problem loss trainable param with noise",weight)

    def get_equation(self):
        return Second2ProblemLossWithNoise(self.get_space_range().split())