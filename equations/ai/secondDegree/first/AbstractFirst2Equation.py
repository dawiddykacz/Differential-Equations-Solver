import numpy
import tensorflow

from objects.Equation import *


def random_0_01(shape=()):
    return tensorflow.random.uniform(
        shape=shape,
        minval=-0.1,
        maxval=0.1,
        dtype=tensorflow.float64
    )


pi = tensorflow.constant(numpy.pi, dtype=tensorflow.float64)
w1 = random_0_01()
w2 = random_0_01()
w3 = random_0_01()
w4 = random_0_01()
w5 = random_0_01()


def exact_solution(x):
    return tensorflow.sin(x * pi)


def condition_bc(function, x, noise):
    return tensorflow.square(function(x) - exact_solution(x) + noise)


class AbstractFirst2Problem(Equation):
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
            global w1, w2, w3, w4, w5
            self.__w1 = w1
            self.__w2 = w2
            self.__w3 = w3
            self.__w4 = w4
            self.__w5 = w5
        else:
            self.__w1 = 0
            self.__w2 = 0
            self.__w3 = 0
            self.__w4 = 0
            self.__w5 = 0

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

        bc_1 = condition_bc(function=function, x=one * -1, noise=self.__w1)
        bc_2 = condition_bc(function=function, x=one, noise=self.__w2)
        bc_3 = condition_bc(function=function, x=zero, noise=self.__w3)
        bc_4 = condition_bc(function=function, x=one / 2, noise=self.__w4)
        bc_5 = condition_bc(function=function, x=one / -2, noise=self.__w5)
        return bc_1 + bc_2 + bc_3 + bc_4 + bc_5


class ExactSolution(Function):
    def calculate(self, *vars):
        global pi

        x = vars[0]
        return exact_solution(x)
