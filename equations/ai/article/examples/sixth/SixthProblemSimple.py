import numpy
import tensorflow as tf

from equations.ai.article.examples.sixth.SixthProblem import *


class SixthProblemSimple(SixthProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()))


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]
        one_y = tensorflow.zeros((len(y), 1), dtype=tensorflow.float64)

        with tensorflow.GradientTape(persistent=True) as g:
            for point in vars:
                g.watch(point)
            z = self._ai_solver.calculate(x,one_y)

            differential_x, differential_y = g.gradient(z, [x, y])

        if differential_y is None:
            differential_y = tensorflow.zeros_like(y)
        del g

        b = y * (2 * x * numpy.sin(numpy.pi) - (x * 2 * numpy.sin(numpy.pi)))
        return b+(x*(1-x)*y* (self._ai_solver.calculate(x,y) - self._ai_solver.calculate(x,one_y) - differential_y))