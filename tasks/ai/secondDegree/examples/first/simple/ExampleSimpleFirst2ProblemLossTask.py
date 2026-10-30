from equations.ai.secondDegree.examples.first.simple.ExampleSimpleFirst2ProblemLoss import \
    ExampleSimpleFirst2ProblemLoss
from objects.TaskData import *


class ExampleSimpleFirst2ProblemLossTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(-1, 1)), f"1 example simple problem loss")

    def get_equation(self):
        return ExampleSimpleFirst2ProblemLoss(self.get_space_range().split())

    def get_plot_title(self):
        return f"1 example simple problem loss"
