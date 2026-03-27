import numpy

from objects.Equation import *


class Fourth2Equation(Equation):
    def __init__(self, solution: AISolution):
        super().__init__(solution, ExactSolution(), "cos")


class SolutionFunction(AISolution):
    def __init__(self, space: Space, loss_function: LossFunction, t: TrainableVariables):
        super().__init__(space, loss_function, t)
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]
        return self._ai_solver.calculate(x, y)


class Loss(LossFunction):
    def __init__(self, t: TrainableVariables):
        self.__t = t

    def _left_side_of_the_equation(self, function, *x):
        with tensorflow.GradientTape(persistent=True) as g:
            for point in x:
                g.watch(point)
            z = function(*x)
            y = x[1]
            x = x[0]
            differential_x, differential_y = g.gradient(z, [x, y])

        if differential_x is None:
            differential_x = tensorflow.zeros_like(x)
        differential_x2 = g.gradient(differential_x, x)
        if differential_x2 is None:
            differential_x2 = tensorflow.zeros_like(x)

        if differential_y is None:
            differential_y = tensorflow.zeros_like(y)
        differential_y2 = g.gradient(differential_y, y)
        if differential_y2 is None:
            differential_y2 = tensorflow.zeros_like(y)
        del g

        return self.__t.get_variables()[0] * (differential_x2 + differential_y2)

    def _right_side_of_the_equation(self, function, *x):
        y = x[1]
        x = x[0]

        pi = tensorflow.constant(numpy.pi, dtype=tensorflow.float64)
        return -(pi ** 2) * tensorflow.sin(pi * x) * tensorflow.cos(pi * y)

    def _condition(self, function, *x):
        y = x[1]
        x = x[0]

        zero = tensorflow.zeros_like(x, dtype=tensorflow.float64)
        one = tensorflow.ones_like(x, dtype=tensorflow.float64)
        one_y = tensorflow.ones_like(y, dtype=tensorflow.float64)

        pi = tensorflow.constant(numpy.pi, dtype=tensorflow.float64)

        return abs(function(-1 * one, y) - zero) + abs(function(one, y) - zero) + abs(
            function(x, one_y * -1) + tensorflow.sin(pi * x)) + abs(function(x, one_y) + tensorflow.sin(pi * x))


class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        return tensorflow.sin(numpy.pi * x) * tensorflow.cos(numpy.pi * y)
