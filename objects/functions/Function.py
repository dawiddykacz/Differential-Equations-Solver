from abc import abstractmethod

from objects.space.Space import Space
import tensorflow
import numpy

class Function:
    def calculate_as_numpy(self,space:Space):
        points = space.get_points()
        result = self.calculate(*points)

        if space.get_dimension() == 1:
            return result.numpy().copy()
        return result.numpy().copy().reshape(space.get_shape())

    @abstractmethod
    def calculate(self,*vars):
        pass