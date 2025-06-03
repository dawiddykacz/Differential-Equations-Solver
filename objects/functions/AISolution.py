from solvers.AISolver import AISolver
from objects.functions.Function import *
from objects.functions.loss.LossFunction import LossFunction
from objects.space.Space import Space
from objects.TrainableVariables import TrainableVariables

class AISolution(Function):
    def __init__(self,space:Space,loss_function:LossFunction,trainable_variables:TrainableVariables
    = TrainableVariables()):
        self._ai_solver = AISolver(space,self.calculate,loss_function,trainable_variables)

    def calculate(self,*vars):
        return self._ai_solver.calculate(*vars)

    def solve(self,epochs:int):
        return self._ai_solver.solve(epochs)

    def get_loss_array(self):
        return self._ai_solver.get_loss_array()

    def get_trainable_variables_array(self):
        return self._ai_solver.get_trainable_variables_array()