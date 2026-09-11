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
one = tensorflow.constant(1.0, dtype=tensorflow.float64)
zero = tensorflow.constant(0.0, dtype=tensorflow.float64)

def exact_solution(x, y):
    return tensorflow.sin(x * pi) * tensorflow.cos(y * pi)


def condition_bc(function, x, y, noise=0):
    return tensorflow.square(function(x, y) - exact_solution(x, y) + noise)


class AbstractExampleSecond2Equation(Equation):
    def __init__(self, solution: AISolution):
        super().__init__(solution, ExactSolution(), "1/2*(d^f(x,y)/d^x+*d^f(x,y)/d^y="
                                                    "-pi^2*sin(pi*x)*cos(pi*x)")


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        n = super().calculate(x, y)
        ansatz = -tensorflow.sin(pi * x)
        return (x ** 2 - one) * (y ** 2 - one) * n + ansatz


class Loss(LossFunction):
    def __init__(self, t: TrainableVariables, with_noise: bool):
        self.__t = t

        if with_noise:
            w = [-0.082, 0.016, 0.048, -0.066, 0.042, 0.024, 0.052, 0.067, -0.056, -0.082, 0.089, 0.097, 0.082, 0.020,
                 0.086, 0.095, -0.042, 0.043, -0.048, -0.030, -0.063, -0.064, -0.043, -0.025, -0.031]
        else:
            w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0]
        self.__w = []
        for v in w:
            self.__w.append(tensorflow.constant(v, dtype=tensorflow.float64))

    def _left_side_of_the_equation(self, function, *x):
        x_var = x[0]
        y_var = x[1]

        with tensorflow.GradientTape(persistent=True) as tape2:
            tape2.watch(x_var)
            tape2.watch(y_var)

            with tensorflow.GradientTape(persistent=True) as tape1:
                tape1.watch(x_var)
                tape1.watch(y_var)
                z = function(x_var, y_var)

            differential_x = tape1.gradient(z, x_var)
            differential_y = tape1.gradient(z, y_var)

        differential_x2 = tape2.gradient(differential_x, x_var)
        differential_y2 = tape2.gradient(differential_y, y_var)

        del tape1
        del tape2

        if differential_x2 is None:
            differential_x2 = tensorflow.zeros_like(x_var)
        if differential_y2 is None:
            differential_y2 = tensorflow.zeros_like(y_var)

        a = self.__t.get_variables()[0]

        return a * (differential_x2 + differential_y2) / (pi ** 2)

    def _right_side_of_the_equation(self, function, *x):
        y = x[1]
        x = x[0]

        return -tensorflow.sin(pi * x) * tensorflow.cos(pi * y)

    def _condition_data(self, function, *x):
        zero_x = tensorflow.zeros_like(x[0], dtype=tensorflow.float64)
        one_x = tensorflow.ones_like(x[0], dtype=tensorflow.float64)
        zero_y = tensorflow.zeros_like(x[1], dtype=tensorflow.float64)
        one_y = tensorflow.ones_like(x[1], dtype=tensorflow.float64)

        data_points_x = [-one_x, -one_x / 2, zero_x, one_x / 2, one_x]
        data_points_y = [-one_y, -one_y / 2, zero_y, one_y / 2, one_y]

        bc = 0
        i = 0
        for x1 in data_points_x:
            for y1 in data_points_y:
                bc += condition_bc(function=function, x=x1, y=y1, noise=self.__w[i])
                i += 1

        return bc


class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        return exact_solution(x, y)
