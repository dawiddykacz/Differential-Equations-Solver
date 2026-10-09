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

        return super().calculate(x, y)


class Loss(LossFunction):
    def __init__(self, t: TrainableVariables, with_noise: bool):
        self.__t = t

        if with_noise:
            w = [-0.082, 0.016, 0.048, -0.066, 0.042, 0.024,
                 0.052, 0.067, -0.056]
        else:
            w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._w = tensorflow.constant(w, dtype=tensorflow.float64)
        self._w = tensorflow.reshape(self._w, [-1, 1])

        factors = [-0.9, 0.0, 0.9]
        grid_x, grid_y = numpy.meshgrid(factors, factors, indexing='ij')

        self._bc_x = tensorflow.constant(grid_x.reshape(-1, 1), dtype=tensorflow.float64)
        self._bc_y = tensorflow.constant(grid_y.reshape(-1, 1), dtype=tensorflow.float64)

    @tensorflow.function
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

    @tensorflow.function
    def _right_side_of_the_equation(self, function, *x):
        y = x[1]
        x = x[0]

        return -tensorflow.sin(pi * x) * tensorflow.cos(pi * y)

    @tensorflow.function
    def _condition_data(self, function, *x):
        predictions = function(self._bc_x, self._bc_y)

        exact = exact_solution(self._bc_x, self._bc_y)

        errors = tensorflow.square(predictions - exact + self._w)

        return tensorflow.reduce_sum(errors)


class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        return exact_solution(x, y)
