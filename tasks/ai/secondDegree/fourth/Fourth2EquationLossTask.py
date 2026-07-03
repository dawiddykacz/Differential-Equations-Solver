from objects.TaskData import *
from equations.ai.secondDegree.fourth.Fourth2EquationLoss import Fourth2EquationLoss


class Fourth2EquationLossTask(TaskData):
    def __init__(self,weight:float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1),Range(-1,1)),"4 second equation loss trainable param",weight=weight)

    def get_equation(self):
        return Fourth2EquationLoss(self.get_space_range().split(), weight=self.get_weight())