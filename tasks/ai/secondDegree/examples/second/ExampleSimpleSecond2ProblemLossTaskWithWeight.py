from objects.TaskData import *
from equations.ai.secondDegree.examples.second.ExampleSimpleSecond2EquationLossWithWeight import \
    ExampleSimpleSecond2EquationLossWithWeight


class ExampleSimpleSecond2ProblemLossTaskWithWeightTask(TaskData):
    def __init__(self, with_noise: bool, alpha: float = 0.1, weight_data: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1), Range(-1, 1)),
                         f"2 second simple problem loss (wang and"
                         f" weight_data {weight_data})"
                         f" with noise = {with_noise} alpha = {alpha}")
        self.with_noise = with_noise
        self.alpha = alpha
        self.weight_data = weight_data

    def get_equation(self):
        return ExampleSimpleSecond2EquationLossWithWeight(self.get_space_range().split(), with_noise=self.with_noise,
                                                          alpha=self.alpha, weight_data=self.weight_data)

    def get_plot_title(self):
        return f"2 example simple problem loss (weight)"
