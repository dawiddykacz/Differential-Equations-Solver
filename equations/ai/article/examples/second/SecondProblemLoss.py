from equations.ai.article.examples.second.SecondProblem import *


class SecondProblemLoss(SecondProblem):
    def __init__(self, space: Space):
        super().__init__(SolutionFunction(space, LossSimple()))


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return self._ai_solver.calculate(x)


class LossSimple(Loss):
    def _condition(self, function, *x):
        x = tensorflow.zeros((len(x[0]), 1), dtype=tensorflow.float64)
        return function(x) - 1

    def _condition_weight(self):
        return 10
