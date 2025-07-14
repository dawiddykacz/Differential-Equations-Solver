from equations.ai.article.examples.third.ThirdProblem import *


class ThirdProblemSimple(ThirdProblem):
    def __init__(self, space: Space,weight:float = 10):
        super().__init__(SolutionFunction(space, Loss()))


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return x + x**2 * self._ai_solver.calculate(x)
