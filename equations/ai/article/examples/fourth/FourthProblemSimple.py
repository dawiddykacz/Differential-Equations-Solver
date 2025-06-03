from equations.ai.article.examples.fourth.FourthProblem import *


class FourthProblemSimple(FourthProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()))


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        a = x* (1-x) *y * (1-y) * self._ai_solver.calculate(x,y)
        b = (1 - x) * y ** 3
        c = x * (1 + y ** 3) * numpy.exp(-1)

        d1 = (1+x) * numpy.exp(-1) - (1 - x - 2 * x * numpy.exp(-1))
        d1 = numpy.exp(-x) + numpy.exp(-1) + y* d1

        d = (1 -y) * x * d1

        return a+b+c+d
