from equations.ai.article.examples.thirdb.ThirdbProblem import *


class ThirdbProblemSimple(ThirdbProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()))


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return x * numpy.sin(1)*numpy.exp(-1/5) + x * (1 - x) * self._ai_solver.calculate(x)