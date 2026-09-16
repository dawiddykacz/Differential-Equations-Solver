from objects.TaskData import *
from equations.ai.secondDegree.examples.first.ExampleSimpleFirst2ProblemLoss import ExampleSimpleFirst2ProblemLoss


class ExampleSimpleFirst2ProblemLossTask(TaskData):
    def __init__(self, with_noise: bool, weight_data: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1)), f"1 example simple problem loss "
                                                        f"with noise {with_noise},"
                                                        f" weight_data {weight_data}")
        self.with_noise = with_noise
        self.weight_data = weight_data

    def get_equation(self):
        return ExampleSimpleFirst2ProblemLoss(self.get_space_range().split(), self.with_noise,
                                              weight_data=self.weight_data)

    def get_plot_title(self):
        return f"1 example simple problem loss"
