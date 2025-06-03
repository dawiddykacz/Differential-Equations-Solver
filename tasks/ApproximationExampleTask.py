from objects.TaskData import *
from approximation.ApproximationExample import ApproximationExample


class ApproximationExampleTask(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(0, 1)),"<0;1>")

    def get_equation(self):
        return ApproximationExample(self.get_space_range().split())
