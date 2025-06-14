from equations.ai.article.examples.second.SecondProblem import *


class SecondProblemSimple(SecondProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()))


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return x * self._ai_solver.calculate(x)
