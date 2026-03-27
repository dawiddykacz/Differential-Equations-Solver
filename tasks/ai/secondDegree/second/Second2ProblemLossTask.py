from objects.TaskData import *
from equations.ai.secondDegree.second.Second2EquationLoss import Second2ProblemLoss


class Second2ProblemLossTask(TaskData):
    def __init__(self,weight:float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1)),"2 second problem loss trainable param",weight)

    def get_equation(self):
        return Second2ProblemLoss(self.get_space_range().split())