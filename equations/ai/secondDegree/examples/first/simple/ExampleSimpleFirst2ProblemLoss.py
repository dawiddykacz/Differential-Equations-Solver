from equations.ai.secondDegree.examples.first.AbstractExampleFirst2Equation import *


class ExampleSimpleFirst2ProblemLoss(AbstractExampleFirst2Problem):
    def __init__(self, space: Space):
        super().__init__(SimpleSolutionFunction(space, loss_function=LossSimple()))


class SimpleSolutionFunction(SolutionFunction):
    def calculate(self, *vars):
        x = vars[0]
        return super().calculate(*vars) * x


class LossSimple(Loss):

    def _left_side_of_the_equation(self, function, *x):
        return super()._left_side_of_the_equation(function, *x) / 2
