from objects.TaskData import *
from equations.ai.secondDegree.first.First2EquationLoss import First2ProblemLoss


class First2ProblemLossTask(TaskData):
    def __init__(self,weight:float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1)),"1 second problem loss",weight)

    def get_equation(self):
        return First2ProblemLoss(self.get_space_range().split())