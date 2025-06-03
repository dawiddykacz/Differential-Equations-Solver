from objects.Equation import *


class ApproximationExample(Equation):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, Loss()), ExactSolution(),"approximation y(x)=x^2")


class Loss(LossFunction):
    def _left_side_of_the_equation(self, function, *x):
        y = function(*x)
        x = x[0]
        return y - x**2


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        return self._ai_solver.calculate(vars[0])

class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        return x**2