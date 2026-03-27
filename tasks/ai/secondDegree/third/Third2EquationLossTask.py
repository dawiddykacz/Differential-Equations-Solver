from objects.TaskData import *
from equations.ai.secondDegree.third.Third2EquationLoss import Third2EquationLoss


class Third2EquationLossTask(TaskData):
    def __init__(self,weight:float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1),Range(-1,1)),"3 second equation",weight=weight)

    def get_equation(self):
        return Third2EquationLoss(self.get_space_range().split(), weight=self.get_weight())