from objects.TaskData import *
from equations.ai.secondDegree.examples.first.ExampleFirst2ProblemLossWithWeight import \
    ExampleFirst2EquationLossWithWeight


class ExampleFirst2ProblemLossWithWeightTask(TaskData):
    def __init__(self, with_noise: bool, alpha: float = 0.1, weight_pde: float = 1, weight_conditions: float = 1,
                 weight_data: float = 1):
        super().__init__(SpaceRanges(10, Range(-1, 1)), f"1 example problem loss (wang and"
                                                        f" weight_pde {weight_pde}"
                                                        f" weight_conditions {weight_conditions},"
                                                        f" weight_data {weight_data})"
                                                        f" with noise = {with_noise} alpha = {alpha}")
        self.with_noise = with_noise
        self.alpha = alpha
        self.weight_conditions = weight_conditions
        self.weight_data = weight_data
        self.weight_pde = weight_pde

    def get_equation(self):
        return ExampleFirst2EquationLossWithWeight(self.get_space_range().split(), self.with_noise, self.alpha,
                                                   weight_pde=self.weight_pde,
                                                   weight_conditions=self.weight_conditions,
                                                   weight_data=self.weight_data)

    def get_plot_title(self):
        return f"1 example problem loss (weight)"
