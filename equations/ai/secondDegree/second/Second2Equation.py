import numpy

from objects.Equation import *


class Second2Problem(Equation):
    def __init__(self,solution:AISolution):
        super().__init__(solution, ExactSolution(),"example")


class Loss(LossFunction):
    def __init__(self, t: TrainableVariables):
        self.__t = t
    def _left_side_of_the_equation(self, function, *x):
        with tensorflow.GradientTape(persistent=True) as g:
            for point in x:
                g.watch(point)
            y = function(*x)
            x = x[0]
            differential = g.gradient(y, x)

        if differential is None:
            differential = tensorflow.zeros_like(x)
        differential2 = g.gradient(differential, x)
        if differential2 is None:
            differential2 = tensorflow.zeros_like(x)
        del g

        return differential2 * self.__t.get_variables()[0]

    def _right_side_of_the_equation(self, function, *x):
        x = x[0]
        pi = tensorflow.constant(numpy.pi, dtype = tensorflow.float64)
        return - (pi ** 2 * tensorflow.sin(x*pi)) / 2

class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        return tensorflow.sin(x * tensorflow.constant(numpy.pi, dtype = tensorflow.float64))