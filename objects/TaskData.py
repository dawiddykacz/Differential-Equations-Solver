from objects.Range import Range
from objects.space.SpaceRanges import SpaceRanges
from equations.ExampleEquation import *


class TaskData:
    def __init__(self, space_range: SpaceRanges, name: str = ""):
        self.__space_range = space_range
        self.__name = name

    @abstractmethod
    def get_equation(self):
        pass

    def get_space_range(self):
        return self.__space_range

    def get_name(self):
        return f"{self.__name} {self.get_equation().get_name()}"

    def get_task_name(self):
        name = self.__name.split(" ")
        return f'{name[0]} {name[1]}'