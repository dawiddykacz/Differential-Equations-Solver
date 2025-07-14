from equations.ai.article.examples.third.ThirdProblem import *

w = 1


class ThirdProblemLoss(ThirdProblem):
    def __init__(self, space: Space, weight: float = 10):
        super().__init__(SolutionFunction(space, Loss()))
        w = weight


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return self._ai_solver.calculate(x)


class LossSimple(Loss):
    def _condition(self, function, *x):
        x = tensorflow.zeros((len(x[0]), 1), dtype=tensorflow.float64)
        return function(x) - 1

    def _condition_weight(self):
        return w
