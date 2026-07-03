from equations.ai.secondDegree.first.First2Equation import *

w = 1


class First2ProblemLoss(First2Problem):
    def __init__(self, space: Space, solution: AISolution = None, weight: float = 1):
        if solution is None:
            super().__init__(SolutionFunction(space, LossSimple()))
        else:
            super().__init__(solution)
        global w
        w = weight


class SolutionFunction(AISolution):
    def calculate(self, *vars):
        x = vars[0]
        return self._ai_solver.calculate(x)


class LossSimple(Loss):
    def _condition(self, function, *x):
        zero = tensorflow.zeros_like(x[0], dtype=tensorflow.float64)
        one = tensorflow.ones_like(x[0], dtype=tensorflow.float64)

        return tensorflow.abs(function(one * -1) - zero) + tensorflow.abs(function(one) - zero)

    def _condition_weight(self):
        return w
