import numpy
import tensorflow

from objects.Equation import *


pi = tensorflow.constant(numpy.pi, dtype=tensorflow.float64)


def exact_solution(x):
    return tensorflow.sin(x * pi)


def condition_bc(function, x, noise):
    return tensorflow.square(function(x) - exact_solution(x) + noise)


class AbstractExampleFirst2Problem(Equation):
    def __init__(self, solution: AISolution):
        super().__init__(solution, ExactSolution(), "d^2f(x)/d^2x= -(pi^2*sin(pi*x))")

class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return self._ai_solver.calculate(x)


class Loss(LossFunction):
    def __init__(self, t: TrainableVariables, with_noise: bool):
        self.__t = t

        if with_noise:
            w = [-0.082, 0.016, 0.048, -0.066, 0.042]
        else:
            w = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.__w = tensorflow.constant(w, dtype=tensorflow.float64)

        self.__points = tensorflow.constant([-1.0, -0.5, 1.0, 0.5, 0.1], dtype=tensorflow.float64)

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
        global pi

        x = x[0]
        return - (pi ** 2 * tensorflow.sin(x * pi)) / 2

    def _condition_data(self, function, *x):
        base_x = tensorflow.ones_like(x[0], dtype=tensorflow.float64)

        results = []
        for i in range(5):
            point_val = self.__points[i] * base_x
            noise_val = self.__w[i]

            results.append(condition_bc(function, point_val, noise_val))

        return tensorflow.add_n(results)


class ExactSolution(Function):
    def calculate(self, *vars):
        global pi

        x = vars[0]
        return exact_solution(x)
