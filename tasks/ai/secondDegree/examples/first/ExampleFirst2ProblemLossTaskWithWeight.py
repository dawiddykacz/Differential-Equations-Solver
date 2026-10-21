from objects.TaskData import *
from equations.ai.secondDegree.examples.first.ExampleFirst2ProblemLossWithWeight import \
    ExampleFirst2EquationLossWithWeight


def get_wang_title(is_calculating_wang_grads):
    if is_calculating_wang_grads:
        return "wang fix grads"
    return "wang"


class ExampleFirst2ProblemLossWithWeightTask(TaskData):
    def __init__(self, with_noise: bool, alpha: float = 0.1, weight_pde: float = 1, weight_conditions: float = 1,
                 weight_data: float = 1, eps: float = 1e-4,
                 is_calculating_wang_grads: bool = False):
        super().__init__(SpaceRanges(10, Range(-1, 1)), f"1 example problem loss "
                                                        f"({get_wang_title(is_calculating_wang_grads)} and"
                                                        f" weight_pde {weight_pde}"
                                                        f" weight_conditions {weight_conditions},"
                                                        f" weight_data {weight_data})"
                                                        f" with noise = {with_noise} alpha = {alpha}")
        self.with_noise = with_noise
        self.alpha = alpha
        self.weight_conditions = weight_conditions
        self.weight_data = weight_data
        self.weight_pde = weight_pde
        self.eps = eps
        self.is_calculating_wang_grads = is_calculating_wang_grads

    def get_equation(self):
        return ExampleFirst2EquationLossWithWeight(self.get_space_range().split(), self.with_noise, self.alpha,
                                                   weight_pde=self.weight_pde,
                                                   weight_conditions=self.weight_conditions,
                                                   weight_data=self.weight_data,
                                                   epsilon=self.eps,
                                                   is_calculating_wang_grads=self.is_calculating_wang_grads)

    def get_plot_title(self):
        return f"1 example problem loss (weight)"
