from equations.ai.secondDegree.examples.second.basic.ExampleSecond2ProblemLoss import ExampleSecond2ProblemLoss
from objects.TaskData import *


class ExampleSecond2ProblemLossTask(TaskData):
    def __init__(self, weight_pde: float = 1, weight_conditions: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1), Range(-1, 1)),
                         f"2 example problem loss "
                         f"weight_pde {weight_pde} "
                         f"weight_conditions {weight_conditions}")
        self.weight_conditions = weight_conditions
        self.weight_pde = weight_pde

    def get_equation(self):
        return ExampleSecond2ProblemLoss(self.get_space_range().split(),
                                         weight_pde=self.weight_pde,
                                         weight_conditions=self.weight_conditions)

    def get_plot_title(self):
        return f"2 example problem loss"
