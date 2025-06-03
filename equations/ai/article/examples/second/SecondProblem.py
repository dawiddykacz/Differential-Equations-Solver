from objects.Equation import *


class SecondProblem(Equation):
    def __init__(self,solution:AISolution):
        super().__init__(solution, ExactSolution(),"df(x)/dx + 1/5 * f(x) = e^-(x/5) * cos(x) ; f(0) = 0")


class Loss(LossFunction):
    def _left_side_of_the_equation(self, function, *x):
        with tensorflow.GradientTape() as g:
            for point in x:
                g.watch(point)
            y = function(*x)

        x = x[0]
        differential = g.gradient(y, x)
        if differential is None:
            differential = tensorflow.zeros_like(x)

        return differential + y / 5

    def _right_side_of_the_equation(self, function, *x):
        x = x[0]
        return numpy.exp(-x / 5) * numpy.cos(x)

    def _condition_weight(self):
        return 2

class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        return tensorflow.exp(-x/5)*tensorflow.sin(x)