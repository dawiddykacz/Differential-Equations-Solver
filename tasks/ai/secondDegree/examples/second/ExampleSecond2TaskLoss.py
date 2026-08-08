from objects.TaskData import *
from equations.ai.secondDegree.examples.second.ExampleSecond2EquationLoss import ExampleSecond2EquationLoss


class ExampleSecond2ProblemLossTask(TaskData):
    def __init__(self, with_noise: bool, weight: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1), Range(-1, 1)),
                         f"2 second problem loss with noise {with_noise}", weight=weight)
        self.with_noise = with_noise

    def get_equation(self):
        return ExampleSecond2EquationLoss(self.get_space_range().split(), with_noise=self.with_noise,
                                          weight=self.get_weight())

    def get_plot_title(self):
        return f"1 example problem loss n = {self.get_weight()}"
