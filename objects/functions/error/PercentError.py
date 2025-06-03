from objects.functions.Function import Function
from objects.functions.AISolution import AISolution
import numpy


class PercentError(Function):
    def __init__(self, solution: AISolution, exact_solution: Function):
        self.__solution = solution
        self.__exact_solution = exact_solution

    def calculate(self, *vars):
        if 0 in self.__exact_solution.calculate_as_numpy(*vars):
            return None
        y = ((self.__exact_solution.calculate_as_numpy(*vars) - self.__solution.calculate_as_numpy(*vars)) * 100 /
             self.__exact_solution.calculate_as_numpy(*vars))
        return numpy.absolute(y)
