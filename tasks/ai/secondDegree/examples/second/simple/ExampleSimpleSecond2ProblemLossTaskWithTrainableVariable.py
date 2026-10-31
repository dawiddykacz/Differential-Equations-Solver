from objects.TaskData import *
from equations.ai.secondDegree.examples.second.simple.ExampleSimpleSecond2EquationLossWithTrainableVariable import \
    ExampleSimpleSecond2EquationLossWithTrainableVariable


class ExampleSimpleSecond2ProblemLossTaskWithTrainableVariable(TaskData):
    def __init__(self, with_noise: bool, weight_pde: float = 1, weight_data: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1), Range(-1, 1)),
                         f"2 example simple problem loss  "
                         f"with noise {with_noise},"
                         f" weight_pde {weight_pde}"
                         f" weight_data {weight_data}")
        self.with_noise = with_noise
        self.weight_pde = weight_pde
        self.weight_data = weight_data

    def get_equation(self):
        return ExampleSimpleSecond2EquationLossWithTrainableVariable(self.get_space_range().split(),
                                                                     with_noise=self.with_noise,
                                                                     weight_pde=self.weight_pde,
                                                                     weight_data=self.weight_data)

    def get_plot_title(self):
        return f"2 example simple problem loss"
