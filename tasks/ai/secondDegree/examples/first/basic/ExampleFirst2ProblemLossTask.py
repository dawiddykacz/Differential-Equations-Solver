from equations.ai.secondDegree.examples.first.basic.ExampleFirst2ProblemLoss import ExampleFirst2ProblemLoss
from objects.TaskData import *


class ExampleFirst2ProblemLossTask(TaskData):
    def __init__(self, weight_pde: float = 1, weight_conditions: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1)), f"1 example problem loss "
                                                        f"weight_pde {weight_pde} "
                                                        f"weight_conditions {weight_conditions}")
        self.weight_conditions = weight_conditions
        self.weight_pde = weight_pde

    def get_equation(self):
        return ExampleFirst2ProblemLoss(self.get_space_range().split(),
                                        weight_pde=self.weight_pde,
                                        weight_conditions=self.weight_conditions)

    def get_plot_title(self):
        return f"1 example problem loss"
