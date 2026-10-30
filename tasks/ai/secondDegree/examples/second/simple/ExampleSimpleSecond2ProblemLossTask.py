from equations.ai.secondDegree.examples.second.simple.ExampleSimpleSecond2ProblemLoss import \
    ExampleSimpleSecond2ProblemLoss
from objects.TaskData import *


class ExampleSimpleSecond2ProblemLossTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(-1, 1), Range(-1, 1)),
                         f"2 example simple problem loss")

    def get_equation(self):
        return ExampleSimpleSecond2ProblemLoss(self.get_space_range().split())

    def get_plot_title(self):
        return f"2 example simple problem loss"
