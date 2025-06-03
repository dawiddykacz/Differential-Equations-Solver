from equations.ai.article.examples.first.FirstProblem import *


class FirstProblemSimple(FirstProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()))


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return x * self._ai_solver.calculate(x) + 1
