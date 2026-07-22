import numpy

from objects.Equation import *


def random_0_01(shape=()):
    return tensorflow.random.uniform(
        shape=shape,
        minval=-0.1,
        maxval=0.1,
        dtype=tensorflow.float64
    )


pi = tensorflow.constant(numpy.pi, dtype=tensorflow.float64)


def exact_solution(x, y):
    return tensorflow.sin(x * pi) * tensorflow.cos(y * pi)


def condition_bc(function, x, y, noise=0):
    return tensorflow.square(function(x, y) - exact_solution(x, y) + noise)


class AbstractExampleSecond2Equation(Equation):
    def __init__(self, solution: AISolution):
        super().__init__(solution, ExactSolution(), "1/2*(d^f(x,y)/d^x+*d^f(x,y)/d^y="
                                                    "-pi^2*sin(pi*x)*cos(pi*x)")


class Loss(LossFunction):
    def __init__(self, t: TrainableVariables, with_noise: bool):
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

        return (differential_x2 + differential_y2) * self.__t.get_variables()[0]

    def _right_side_of_the_equation(self, function, *x):
        y = x[1]
        x = x[0]

        return -(pi ** 2) * tensorflow.sin(pi * x) * tensorflow.cos(pi * y)

    def _condition(self, function, *x):
        y = x[1]
        x = x[0]

        one = tensorflow.ones_like(x, dtype=tensorflow.float64)
        one_y = tensorflow.ones_like(y, dtype=tensorflow.float64)

        return (condition_bc(function=function, x=one * -1, y=y) + condition_bc(function=function, x=one, y=y) +
                condition_bc(function=function, x=x, y=one_y * -1) + condition_bc(function=function, x=x, y=one_y))

    def _condition_data(self, function, *x):
        zero = tensorflow.zeros_like(x[0], dtype=tensorflow.float64)
        one = tensorflow.ones_like(x[0], dtype=tensorflow.float64)

        arr = [-3 / 4, -1 / 2, -1 / 4, 0, 1 / 4, 1 / 2, 3 / 4]
        bc = condition_bc(function=function, x=zero, y=zero)

        for multiplier_x in arr:
            for multiplier_y in arr:
                bc = condition_bc(function=function, x=one * multiplier_x, y=one * multiplier_y)
        return bc


class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        return exact_solution(x, y)
