from objects.TaskData import *
from equations.ai.secondDegree.examples.first.ExampleFirst2EquationLoss import ExampleFirst2ProblemLoss


class ExampleFirst2ProblemLossTask(TaskData):
    def __init__(self, with_noise: bool, weight: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1)), f"1 second problem loss "
                                                        f"with noise {with_noise}", weight)
        self.with_noise = with_noise

    def get_equation(self):
        return ExampleFirst2ProblemLoss(self.get_space_range().split(),self.with_noise,self.get_weight())
