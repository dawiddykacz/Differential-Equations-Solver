from objects.TaskData import *
from equations.ai.secondDegree.examples.second.ExampleSecond2EquationLossWithWeight import \
    ExampleSecond2EquationLossWithWeight


class ExampleSecond2ProblemLossTaskWithWeight(TaskData):
    def __init__(self, with_noise: bool, alpha: float = 0.1, alpha_lower: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1), Range(-1, 1)),
                         f"2 second problem loss (weight) with noise = {with_noise} "
                         f"alpha = {alpha} alpha_lower = {alpha_lower}")
        self.with_noise = with_noise
        self.alpha = alpha
        self.alpha_lower = alpha_lower

    def get_equation(self):
        return ExampleSecond2EquationLossWithWeight(self.get_space_range().split(), with_noise=self.with_noise,
                                                    alpha=self.alpha, alpha_lower=self.alpha_lower)

    def get_plot_title(self):
        return f"2 example problem loss (weight)"
