from objects.Range import Range
from objects.space.SpaceRanges import SpaceRanges
from equations.ExampleEquation import *


class TaskData:
    def __init__(self, space_range: SpaceRanges, name: str = "",weight: float = None):
        self.__space_range = space_range
        self.__name = name
        self.__weight = weight

    @abstractmethod
    def get_equation(self):
        pass

    def get_space_range(self):
        return self.__space_range

    def get_name(self):
        return f"{self.__name} {self.get_equation().get_name()}"

    def get_task_name(self):
        if self.__weight is not None:
            return f"{self.__name} n = {self.__weight}"
        return self.__name

    def get_task_name_simple(self):
        return self.__name

    def get_weight(self):
        return self.__weight