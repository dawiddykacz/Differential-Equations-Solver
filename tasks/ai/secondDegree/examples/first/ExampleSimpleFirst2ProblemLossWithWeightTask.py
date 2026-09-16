from objects.TaskData import *
from equations.ai.secondDegree.examples.first.ExampleSimpleFirst2EquationLossWithWeight import \
    ExampleSimpleFirst2EquationLossWithWeight


class ExampleSimpleFirst2ProblemLossWithWeightTask(TaskData):
    def __init__(self, with_noise: bool, alpha: float = 0.1, weight_data: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1)), f"1 example simple problem loss "
                                                        f"(wang and"
                                                        f" weight_data {weight_data})"
                                                        f" with noise = {with_noise} alpha = {alpha}")
        self.with_noise = with_noise
        self.alpha = alpha
        self.weight_data = weight_data

    def get_equation(self):
        return ExampleSimpleFirst2EquationLossWithWeight(self.get_space_range().split(), self.with_noise, self.alpha,
                                                         weight_data=self.weight_data)

    def get_plot_title(self):
        return f"1 example simple problem loss (weight)"
