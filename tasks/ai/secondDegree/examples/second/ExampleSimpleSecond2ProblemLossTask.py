from objects.TaskData import *
from equations.ai.secondDegree.examples.second.ExampleSimpleSecond2EquationLoss import ExampleSimpleSecond2EquationLoss


class ExampleSimpleSecond2ProblemLossTask(TaskData):
    def __init__(self, with_noise: bool, weight_data: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1), Range(-1, 1)),
                         f"2 example simple problem loss  "
                         f"with noise {with_noise},"
                         f" weight_data {weight_data}")
        self.with_noise = with_noise
        self.weight_data = weight_data

    def get_equation(self):
        return ExampleSimpleSecond2EquationLoss(self.get_space_range().split(), with_noise=self.with_noise,
                                                weight_data=self.weight_data)

    def get_plot_title(self):
        return f"2 example simple problem loss"
