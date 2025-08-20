import tensorflow
from  helpers.EquationsHelper import ArticleProblemsHelper

from equations.ai.article.examples.sixth.SixthProblem import *


class SixthProblemSimple(SixthProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()))

f0 = lambda x: tensorflow.zeros_like(x)
f1 = lambda x: tensorflow.zeros_like(x)
g0 = lambda x: tensorflow.zeros_like(x)
g1 = lambda x: 2*tensorflow.sin(x*numpy.pi)

b = ArticleProblemsHelper(f0, f1, g0, g1)

class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]
        one_y = tensorflow.ones_like(y, dtype=tensorflow.float64)

        with tensorflow.GradientTape(persistent=True) as g:
            g.watch(one_y)
            z = self._ai_solver.calculate(x,one_y)

            differential_y = g.gradient(z,  one_y)

        if differential_y is None:
            differential_y = tensorflow.zeros_like(one_y)
        del g

        return b.calculate(x,y) + (x*(1-x)*y * (self._ai_solver.calculate(x,y) - self._ai_solver.calculate(x,one_y) - differential_y))