from objects.Equation import *


class FifthProblem(Equation):
    def __init__(self,solution:AISolution):
        super().__init__(solution, ExactSolution(),"cos")


class Loss(LossFunction):
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

        return differential_x2 + differential_y2

    def _right_side_of_the_equation(self, function, *x):
        y = x[1]
        x = x[0]
        return numpy.exp(-x)*(x - 2 + y ** 3 + 6 * y)

class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        return tensorflow.exp(-x)*(x+y**3)