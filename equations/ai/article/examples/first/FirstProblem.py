from objects.Equation import *


class FirstProblem(Equation):
    def __init__(self,solution:AISolution):
        super().__init__(solution, ExactSolution(),"df(x)/dx + (x + (1+3x^2) / (1+x+x^3)) f(x) = x^3 + 2x + x^2 * (1 + 3x^2) / (1 + x + x^3); f(0) = 1")


#df(x)/dx + (x + (1+3x^2) / (1+x+x^3)) f(x) = x^3 + 2x + x^2 * (1 + 3x^2) / (1 + x + x^3);
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

        return differential + (x + (1 + 3*x**2) / (1 + x + x**3) ) * y

    def _right_side_of_the_equation(self, function, *x):
        x = x[0]
        return x**3 + 2 * x + x**2 * (1 + 3*x**2) / (1 + x + x**3)

class ExactSolution(Function):
    def calculate(self, *vars):
        x = vars[0]
        return tensorflow.exp(-x**2/2)/(1+x+x**3)+x**2