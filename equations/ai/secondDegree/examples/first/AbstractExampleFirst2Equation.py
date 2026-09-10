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
        self.__w = []
        for v in w:
            self.__w.append(tensorflow.constant(v, dtype=tensorflow.float64))

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
        zero = tensorflow.zeros_like(x[0], dtype=tensorflow.float64)
        one = tensorflow.ones_like(x[0], dtype=tensorflow.float64)

        bc_1 = condition_bc(function=function, x=one * -1, noise=self.__w[0])
        bc_5 = condition_bc(function=function, x=one / -2, noise=self.__w[1])
        bc_2 = condition_bc(function=function, x=one, noise=self.__w[2])
        bc_4 = condition_bc(function=function, x=one / 2, noise=self.__w[3])
        bc_3 = condition_bc(function=function, x=zero, noise=self.__w[4])
        return bc_1 + bc_2 + bc_3 + bc_4 + bc_5


class ExactSolution(Function):
    def calculate(self, *vars):
        global pi

        x = vars[0]
        return exact_solution(x)
