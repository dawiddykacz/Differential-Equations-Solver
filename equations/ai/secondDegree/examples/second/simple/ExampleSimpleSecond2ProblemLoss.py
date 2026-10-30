from equations.ai.secondDegree.examples.second.AbstractExampleSecond2Equation import *


class ExampleSimpleSecond2ProblemLoss(AbstractExampleSecond2Equation):
    def __init__(self, space: Space):
        super().__init__(SimpleSolutionFunction(space, loss_function=LossSimple()))


class SimpleSolutionFunction(SolutionFunction):
    def calculate(self, *vars):
        x = vars[0]
        y = vars[1]

        n = super().calculate(x, y)
        ansatz = -tensorflow.sin(pi * x)
        return (tensorflow.square(x) - one) * (tensorflow.square(y) - one) * n + ansatz


class LossSimple(Loss):

    def _left_side_of_the_equation(self, function, *x):
        return super()._left_side_of_the_equation(function, *x) / 2
