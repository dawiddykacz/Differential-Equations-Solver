from objects.TaskData import *
from equations.ai.secondDegree.examples.second.ExampleSecond2EquationLossWithWeight import \
    ExampleSecond2EquationLossWithWeight


class ExampleSecond2ProblemLossTaskWithWeight(TaskData):
    def __init__(self, with_noise: bool):
        super().__init__(SpaceRanges(10, Range(-1, 1), Range(-1, 1)),
                         f"2 second problem loss (weight) with noise {with_noise}")
        self.with_noise = with_noise

    def get_equation(self):
        return ExampleSecond2EquationLossWithWeight(self.get_space_range().split(), with_noise=self.with_noise)
