from objects.TaskData import *
from equations.ExampleEquation import ExampleEquation


class Example2Task(TaskData):
    def __init__(self):
        super().__init__(SpaceRanges(10, Range(-1, 10)),"<-1;10>")

    def get_equation(self):
        return ExampleEquation(self.get_space_range().split())
