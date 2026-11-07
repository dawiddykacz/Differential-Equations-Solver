import math
import numpy
import tensorflow

from objects.Equation import *
from equations.ai.secondDegree.examples.WangHelper import calculate_wang
from equations.ai.secondDegree.examples.second.AbstractExampleSecond2Equation import *


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

# Parametry z artykułu Wang et al.
a1 = tensorflow.constant(1.0, dtype=tensorflow.float64)
a2 = tensorflow.constant(4.0, dtype=tensorflow.float64)
k_wave = tensorflow.constant(1.0, dtype=tensorflow.float64)  # Współczynnik k


def exact_solution(x, y):
    # u(x,y) = sin(a1*pi*x)*sin(a2*pi*y)
    return tensorflow.sin(a1 * pi * x) * tensorflow.sin(a2 * pi * y)


def condition_bc(function, x, y, noise=0):
    return tensorflow.square(function(x, y) - exact_solution(x, y) + noise)


class AbstractNavierEquation(Equation):
    def __init__(self, solution: AISolution):
        # Aktualizacja opisu równania Helmholtza
        super().__init__(solution, ExactSolution(), "d^2f/dx^2 + d^2f/dy^2 + k^2*f = q(x,y)")


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]
        return super().calculate(x, y)


class Loss(LossFunction):

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

        # Lewa strona: Delta u + k^2 u
        return differential_x2 + differential_y2 + (k_wave ** 2) * z

    @tensorflow.function
    def _right_side_of_the_equation(self, function, *x):
        x_var = x[0]
        y_var = x[1]

        # Prawa strona (wyraz źródłowy q(x,y))
        term1 = -((a1 * pi) ** 2) * tensorflow.sin(a1 * pi * x_var) * tensorflow.sin(a2 * pi * y_var)
        term2 = -((a2 * pi) ** 2) * tensorflow.sin(a1 * pi * x_var) * tensorflow.sin(a2 * pi * y_var)
        term3 = (k_wave ** 2) * tensorflow.sin(a1 * pi * x_var) * tensorflow.sin(a2 * pi * y_var)

        return term1 + term2 + term3


class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]
        return exact_solution(x, y)