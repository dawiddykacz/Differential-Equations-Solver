from objects.functions.AISolution import *
import tensorflow
import numpy


class Equation:
    def __init__(self, solution_function: AISolution,exact_solution: Function = None,name: str = ""):
        self.solution_function = solution_function
        self.name = name
        self.exact_solution = exact_solution

    def get_solution_function(self):
        return self.solution_function

    def get_exact_solution(self):
        return self.exact_solution

    def get_name(self):
        return self.name
