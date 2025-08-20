import numpy

from objects.Equation import *


class SixthProblem(Equation):
    def __init__(self,solution:AISolution):
        super().__init__(solution, ExactSolution(),"cos")


class Loss(LossFunction):
    def _left_side_of_the_equation(self, function, *x):

        with tensorflow.GradientTape(persistent=True) as g:
            for point in x:
                g.watch(point)
            z = function(*x)
            x, y = x
            differential_x, differential_y = g.gradient(z, [x, y])

        if differential_x is None:
            differential_x = tensorflow.zeros_like(x)
        differential_x2 = g.gradient(differential_x, x)
        if differential_x2 is None:
            differential_x2 = tensorflow.zeros_like(differential_x)

        if differential_y is None:
            differential_y = tensorflow.zeros_like(y)
        differential_y2 = g.gradient(differential_y, y)
        if differential_y2 is None:
            differential_y2 = tensorflow.zeros_like(differential_y)
        del g

        return differential_x2 + differential_y2

    def _right_side_of_the_equation(self, function, *x):
        x, y = x
        return (2 - numpy.pi ** 2 * y ** 2)*numpy.sin(numpy.pi*x)

class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        return y ** 2 * numpy.sin(numpy.pi*x)