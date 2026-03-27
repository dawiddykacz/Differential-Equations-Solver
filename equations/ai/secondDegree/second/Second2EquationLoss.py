from equations.ai.secondDegree.second.Second2Equation import *

w = 1


class Second2ProblemLoss(Second2Problem):
    def __init__(self, space: Space, solution: AISolution = None, weight: float = 1):
        t = TrainableVariables([1])
        if solution is None:
            super().__init__(SolutionFunction(space, LossSimple(t), t))
        else:
            super().__init__(solution)
        global w
        w = weight


class SolutionFunction(AISolution):
    def __init__(self, space: Space, loss_function: LossFunction, t: TrainableVariables):
        super().__init__(space, loss_function, t)

    def calculate(self, *vars):
        x = vars[0]
        return self._ai_solver.calculate(x)


class LossSimple(Loss):
    def _condition(self, function, *x):
        zero = tensorflow.zeros_like(x[0], dtype=tensorflow.float64)
        one = tensorflow.ones_like(x[0], dtype=tensorflow.float64)

        return tensorflow.abs(function(one * -1) - zero) + tensorflow.abs(function(one) - zero) + tensorflow.abs(
            function(zero) - zero) + tensorflow.abs(function(one / 2) - one) + tensorflow.abs(function(-one / 2) + one)
    def _condition_weight(self):
        return w
