from objects.Equation import *


class ExampleEquation(Equation):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()), ExactSolution(),"y'=e^(-x/5)*cos(x)-y/5, y(0)=0")


class Loss(LossFunction):
    def _left_side_of_the_equation(self, function, *x):
        with tensorflow.GradientTape() as g:
            for point in x:
                g.watch(point)
            y = function(*x)
        psi_p = g.gradient(y, x[0])
        if psi_p is None:
            psi_p = tensorflow.zeros_like(x[0])

        x = x[0]
        return psi_p + y / 5 - (numpy.exp(-x / 5) * numpy.cos(x))


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        return vars[0] * self._ai_solver.calculate(vars[0])

class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        return tensorflow.exp(-x/5)*tensorflow.sin(x)