from objects.TaskData import *
from equations.ai.secondDegree.examples.first.ExampleFirst2ProblemLossWithWeight import \
    ExampleFirst2EquationLossWithWeight


class ExampleFirst2ProblemLossWithWeightTask(TaskData):
    def __init__(self, with_noise: bool, alpha: float = 0.1, alpha_lower: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1)), f"1 example problem loss (weight) "
                                                        f"with noise = {with_noise} alpha = {alpha} "
                                                        f"alpha_lower = {alpha_lower}")
        self.with_noise = with_noise
        self.alpha = alpha
        self.alpha_lower = alpha_lower

    def get_equation(self):
        return ExampleFirst2EquationLossWithWeight(self.get_space_range().split(), self.with_noise, self.alpha,
                                                   self.alpha_lower)

    def get_plot_title(self):
        return f"1 example problem loss (weight)"
