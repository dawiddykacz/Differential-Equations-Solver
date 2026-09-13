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
        return (tensorflow.square(x) - one) * (tensorflow.square(y) - one) * n + ansatz


class Loss(LossFunction):
    def __init__(self, t: TrainableVariables, with_noise: bool):
        self.__t = t

        if with_noise:
            w = [-0.082, 0.016, 0.048, -0.066, 0.042, 0.024, 0.052, 0.067, -0.056, -0.082, 0.089, 0.097, 0.082, 0.020,
                 0.086, 0.095, -0.042, 0.043, -0.048, -0.030, -0.063, -0.064, -0.043, -0.025, -0.031]
        else:
            w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0]
        self.__w = tensorflow.constant(w, dtype=tensorflow.float64)

    def _left_side_of_the_equation(self, function, *x):
        x_var = x[0]
        y_var = x[1]

        with tensorflow.GradientTape(persistent=True) as tape2:
            tape2.watch(x_var)
            tape2.watch(y_var)

            with tensorflow.GradientTape() as tape1:
                tape1.watch(x_var)
                tape1.watch(y_var)
                z = function(x_var, y_var)

            differential_x, differential_y = tape1.gradient(z, [x_var, y_var])

        differential_x2 = tape2.gradient(differential_x, x_var)
        differential_y2 = tape2.gradient(differential_y, y_var)

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
        base_x = tensorflow.ones_like(x[0], dtype=tensorflow.float64)
        base_y = tensorflow.ones_like(x[1], dtype=tensorflow.float64)

        factors = [-1.0, -0.5, 0.0, 0.5, 1.0]

        results = []
        i = 0
        for f_x in factors:
            for f_y in factors:
                point_x = f_x * base_x
                point_y = f_y * base_y

                noise_val = self.__w[i]

                results.append(
                    condition_bc(function=function, x=point_x, y=point_y, noise=noise_val)
                )
                i += 1

        return tensorflow.reduce_sum(tensorflow.add_n(results))

class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        return exact_solution(x, y)
