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
w1 = random_0_01() * 0
w2 = random_0_01() * 0


def cos(f, x):
    return tensorflow.square(f(x) - tensorflow.sin(x * pi))

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
            global w1, w2
            self.__w1 = w1
            self.__w2 = w2
        else:
            self.__w1 = 0
            self.__w2 = 0

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

        bc_left = tensorflow.square(function(one * -1) - zero)
        bc_right = tensorflow.square(function(one) - zero)
        return bc_left + bc_right + cos(function,zero) + cos(function,one/2) + cos(function,one/-2)
class ExactSolution(Function):
    def calculate(self, *vars):
        global pi

        x = vars[0]
        return tensorflow.sin(x * pi)
