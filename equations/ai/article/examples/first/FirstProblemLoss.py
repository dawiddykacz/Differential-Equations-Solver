import tensorflow.python.keras.backend

from equations.ai.article.examples.first.FirstProblem import *

w = 10
class FirstProblemLoss(FirstProblem):
    def __init__(self, space: Space,solution:AISolution = None,weight:float = 10):
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

        return function(zero) - one
    def _condition_weight(self):
        return w